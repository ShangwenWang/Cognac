import json
import javalang
from re import finditer
import re
import argparse
import os, fnmatch
import sys, eventlet
from tqdm import tqdm
import multiprocessing
import subprocess
from copy import deepcopy
from collections import defaultdict

excluded = {
    'Separator',
    'Operator',
}

orederMapping = {
    'StatementExpression': 1,
    'LocalVariableDeclaration': 2,
    'AssertStatement': 3,
    'WhileStatement': 4,
    'IfStatement': 5,
    'TryStatement': 6,
    'ThrowStatement': 7,
    'SwitchStatement': 8,
    'SwitchStatementCase': 9,
    'ReturnStatement': 10,
    'DoStatement': 11,
    'ForStatement': 12,
    'FieldDeclaration': 13,
    'ClassName': 14,
    'ReturnType': 15,
    'Caller': 16,
    'Callee': 17,
    'SynchronizedStatement': 18,
    'Parameter': 19,
    'Callee.StatementExpression': 20,
    'Callee.LocalVariableDeclaration': 21,
    'Callee.AssertStatement': 22,
    'Callee.WhileStatement': 23,
    'Callee.IfStatement': 24,
    'Callee.TryStatement': 25,
    'Callee.ThrowStatement': 26,
    'Callee.SwitchStatement': 27,
    'Callee.SwitchStatementCase': 28,
    'Callee.ReturnStatement': 29,
    'Callee.DoStatement': 30,
    'Callee.ForStatement': 31,
    'Callee.FieldDeclaration': 32,
    'Callee.ClassName': 33,
    'Callee.ReturnType': 34,
    'Callee.SynchronizedStatement': 35,
    'Callee.Parameter': 36,
    'Caller.ReturnType': 37,
    'Caller.Parameter': 38,
    '<UNK>': 39
}

frequence = {
    "ReturnStatement": 0.224632945,
    "ReturnType": 0.160275608,
    "Parameter": 0.125718472,
    "StatementExpression": 0.125082106,
    "Callee.ReturnStatement": 0.104886315,
    "LocalVariableDeclaration": 0.099483322,
    "ClassName": 0.096258274,
    "Callee.ReturnType": 0.095125684,
    "AssertStatement": 0.079472415,
    "SwitchStatement": 0.078355678,
    "SynchronizedStatement": 0.075410688,
    "IfStatement": 0.068725314,
    "Callee.StatementExpression": 0.066654503,
    "ForStatement": 0.062350527,
    "Caller.ReturnType": 0.061111598,
    "ThrowStatement": 0.055662937,
    "Callee.LocalVariableDeclaration": 0.055055787,
    "FieldDeclaration": 0.045173916,
    "Callee.Parameter": 0.044037616,
    "Caller.Parameter": 0.039394445,
    "WhileStatement": 0.038646218,
    "Callee.SynchronizedStatement": 0.036989493,
    "Callee.IfStatement": 0.036380593,
    "Callee.ForStatement": 0.035718878,
    "Callee.SwitchStatement": 0.034618791,
    "DoStatement": 0.03426961,
    "SwitchStatementCase": 0.033224994,
    "Callee.ThrowStatement": 0.032856621,
    "Callee.AssertStatement": 0.032127471,
    "Callee.SwitchStatementCase": 0.025606972,
    "Callee.DoStatement": 0.020817933,
    "Callee.WhileStatement": 0.020542936,
    "Callee.FieldDeclaration": 0.020261926,
    "TryStatement": 0.013649739,
    "Callee.TryStatement": 0.005474447
    }

separator = {'(', ',', ')', '}', '{', '[', ']', '.', ';'}
stripAllRegex = re.compile('[\s]+')
memberFinder = re.compile("member=([\S]+),")
qualifierFinder = re.compile("qualifier=([\S]+),")
startrule = re.compile('^[a-zA-z]{1}.*$')
numPattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$|[-+]?\.?[0-9][x]?[a-fA-F\d]*[Ll]?$')
METHOD_SUBTOKEN_SEPARATOR = re.compile(
    r"([a-z]+)([A-Z][a-z]+)|([A-Z][a-z]+)|[_.]([a-z]+)([A-Z][a-z]+)|[._]?([a-z]+)[0-9]*[._]*|[._]([A-Z][a-z]+)|[._]?([A-Z]+)([A-Z][a-z]*)[0-9]*[._]*|([a-z]+)[0-9]+|([a-z]+)([A-Z]+[a-z]*)|[._]([0-9]*)")


def is_number(num):
    result = numPattern.match(num)
    if result:
        return True
    else:
        return False


def createRamDisk(targetPath, size):
    if not os.path.exists(targetPath):
        os.mkdir(targetPath)
    ret = subprocess.run(["mount", "-t", "tmpfs", "-osize=" + size, "tmpfs", targetPath],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if ret.returncode != 0:
        print("Failed to create ramDisk")

def umoutRamDisk(ramDiskPath):
    ret = subprocess.run(["umount", "-v", ramDiskPath], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    if ret.returncode != 0:
        print(ret.stderr)
        print(ret.stdout)
        return -1
    return 0

def moveAndDecompress(filePath, targetDir):
    subprocess.run(["cp", "-r", filePath, targetDir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ret = subprocess.run(["tar", "-axf", os.path.join(targetDir, filePath), '-C', targetDir, ">/dev/null 2>&1"])
    if ret.returncode != 0:
        print("Failed to decompress file")
        return -1
    return 0

def getFileSize(filePath):
    return os.path.getsize(filePath)/1000

def travFolder(dir, files=[], suffix=''):
    ifnormal = False
    try:
        listdirs = os.listdir(dir)
        ifnormal = True
    except:
        print("path error")
    if ifnormal:
        for f in listdirs:
            pattern = '*.' + suffix if suffix != '' else '*.*'
            if os.path.isfile(os.path.join(dir, f)) and fnmatch.fnmatch(f, pattern):
                #and 'test' not in os.path.join(dir, f) and 'src' in os.path.join(dir, f):
                filename = os.path.join(dir, f)
                files.append(filename) if getFileSize(filename) <=128 else None
            elif os.path.isdir(os.path.join(dir, f)):
                travFolder(dir + '/' + f, files, suffix)
    return files

splittedRec = {}

def get_subtokens(method_name):
    if method_name in splittedRec:
        return splittedRec[method_name]
    else:
        returnRec = [x.lower() for x in re.split('[_ \d]',re.sub(r'([A-Z]+[a-z])', r' \1', re.sub(r'([A-Z]+)', r' \1', method_name)).strip()) if x !='']
        splittedRec[method_name] = returnRec
    return returnRec

stripSymbol = re.compile('\W+')



lock = multiprocessing.Lock()


def core(args):
    repoPath, outputPath, position, consideredType = args
    resRec = []
    with eventlet.Timeout(10, False):
        validFiles = travFolder(repoPath, [], suffix='java')
    # with lock:
    #     progress = tqdm(total=validFiles.__len__(), desc=os.path.basename(repoPath), position=position)

    curCallGraph = callGraph(validFiles, consideredType)
    for i, classPath in enumerate(validFiles):
        with eventlet.Timeout(10, False):  
            try:
                extractor = infoExtractor(classPath, consideredType, curCallGraph)
                curRes = extractor.run()
                resRec += curRes
            except:
                # print("Exception happened when extracting information from AST.")
                continue
    try:
        print('Current repo: ' + os.path.basename(repoPath), ':', resRec.__len__())
        saveResult(outputPath, resRec)
    except:
        print('Results are not saved successfully!')


class callGraph:
    def __init__(self, validFiles, consideredType):
        self.validFiles = validFiles
        self.nodes = set()
        for x in validFiles:
            self.getNode(x)
        self.caller = defaultdict(set)
        self.callee = defaultdict(set)
        self.consideredType = consideredType
        self.methodMapping = dict()
        for x in validFiles:
            self.constructGraph(x)


    def getNode(self, filePath):
        with open(filePath, 'r', encoding='utf8', errors='ignore') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('package '):
                package = line[8:].strip('\r\n;')
                node = package + '.' + os.path.basename(filePath).strip('.java')
                self.nodes.add(node)
                break

    @staticmethod
    def findInvocation(AST: str):
        indexes = [x.span() for x in finditer("MethodInvocation", AST)]
        invocations = set()
        for index in indexes:
            bracketCnt = 0
            rightPart = AST[index[1]:]
            validStr = ""
            for i, x in enumerate(rightPart):
                if x == "(":
                    bracketCnt -= 1
                elif x == ")":
                    bracketCnt += 1
                if bracketCnt == -1:
                    validStr += x
                if bracketCnt == 0 and validStr != '':
                    break
            member = [x.strip('\'') for x in memberFinder.findall(validStr)]  # TODO: check it
            qualifier = [x.strip('\'') for x in qualifierFinder.findall(validStr)]
            if member.__len__() == qualifier.__len__() and member.__len__() == 1 and qualifier[0] != 'None':
                invocations.add((qualifier[0], member[0]))
        return invocations

    @staticmethod
    def completeInvocation(localImports, invocations, curClass, superclass, curMethods):
        completedInv = set()
        for x in invocations:
            for y in localImports:
                if x[0] != '' and y.endswith(x[0]):
                    completedInv.add(y + '.' + x[1])
                elif x[0] == '' and x[1] in curMethods:
                    completedInv.add(curClass + '.' + x[1])
                elif x[0] == '' and superclass is not None:
                    completedInv.add(superclass + '.' + x[1])
        return completedInv

    @staticmethod
    def completeSuperclass(localImports, surperclass):
        if isinstance(surperclass, list):
            surperclass = surperclass[0]
        if surperclass is not None:
            for x in localImports:
                if x.endswith(surperclass.name):
                    return x + '.' + surperclass.name
        else:
            return None

    def getClassMethods(self, AST):
        methods = set()
        for method in AST.types[0].body:
            if not isinstance(method, self.consideredType) or method.name == 'main':
                continue
            methods.add(method.name)
        return methods

    def constructGraph(self, filePath):
        with open(filePath, 'r', encoding='utf8', errors='ignore') as f:
            body = f.read()
        try:
            AST = javalang.parse.parse(body)
        except:
            return
        localImports = [x.path for x in AST.imports if x.path in self.nodes]
        if AST.types.__len__() > 0 and AST.types[0] and AST.package:
            curClass = AST.package.name + '.' + AST.types[0].name
        else:
            return
        if hasattr(AST.types[0], 'extends'):
            superclass = self.completeSuperclass(localImports, AST.types[0].extends)
        else:
            superclass = None
        curClassMethods = self.getClassMethods(AST)
        for method in AST.types[0].body:
            #TODO: a question is how to add constructor
            if not isinstance(method, self.consideredType) or method.name == 'main':
                continue
            curMethod = curClass + '.' + method.name
            method.tokens = AST.tokens
            self.methodMapping[curMethod] = method
            invocations = self.findInvocation(str(method))
            invocations_completed = self.completeInvocation(localImports, invocations, curClass, superclass, curClassMethods)
            self.callee[curMethod] = set(invocations_completed)
            for x in invocations_completed:
                self.caller[x].add(curMethod)


    def getCallee(self, method):
        if method in self.callee:
            return self.callee[method]
        else:
            return []

    def getCaller(self, method):
        if method in self.caller:
            return self.caller[method]
        else:
            return []

    def method2AST(self, method):
        if isinstance(method, str):
            if method in self.methodMapping:
                return self.methodMapping[method]
            else:
                return None
        elif isinstance(method, (set, list)):
            calleeAST = []
            for x in method:
                curAST = self.method2AST(x)
                if curAST:
                    calleeAST.append(curAST)
            return calleeAST


class infoExtractor:
    def __init__(self, classPath:str, consideredType, callGraph: callGraph):
        self.classPath = classPath
        # self.callGraph = dict()
        self.callGraph = callGraph
        self.AST = self.getAST()
        self.tokenStreams = dict()
        if self.AST.types.__len__() > 0 and self.AST.package and self.AST.types[0]:
            self.curClass = self.AST.package.name + '.' + self.AST.types[0].name
        else:
            self.curClass = None
        if self.AST is None:
            raise RuntimeError('Failed to generate AST')
        self.consideredType = consideredType

    def getAST(self):
        with open(self.classPath, 'r', encoding='utf8', errors='ignore') as f:
            body = f.read()
        try:
            AST = javalang.parse.parse(body)
        except:
            return None
        return AST

    def findCallee(self, curMethod):
        callee = self.callGraph.getCallee(curMethod)
        calleeAST = self.callGraph.method2AST(callee)
        return calleeAST

    def findCaller(self, curMethod):
        caller = self.callGraph.getCaller(curMethod)
        callerAST = self.callGraph.method2AST(caller)
        return callerAST

    def getTokenStrem(self, method, topk=0):
        methodStr = str(method)
        if method in self.tokenStreams:
            tokenStream = self.tokenStreams[methodStr]
        else:
            tokenStream = [x for x in javalang.ast.get_token_stream_2(method) if x.value != '' and x.value != ' ' and len(
                x.value.strip()) >= 1 and x.curType !="<UNK>"]
            self.tokenStreams[methodStr] = tokenStream
        if topk == 0:
            return tokenStream
        else:
            return sorted(tokenStream, key=lambda x: frequence['Callee.' + x.curType], reverse=True)[:topk]

    def getAllMethod(self, classNode):
        methods = []
        for x in classNode:
            if isinstance(x, self.consideredType):
                methods.append(x)
            elif isinstance(x, javalang.tree.ClassDeclaration):
                methods += self.getAllMethod(x.body)
        return methods

    def run(self):
        if self.AST.types.__len__() == 0:
            return []
        curMethodRes = []
        methods = self.getAllMethod(self.AST.types[0].body)
        for method in methods:
            if not isinstance(method, self.consideredType) or self.curClass is None:
                continue
            curMethodName = self.curClass + '.' + method.name
            #TODO: a question is how to add constructor
            if not isinstance(method, self.consideredType):
                continue
            if method.name == 'main':
                continue
            #define a proxy before splitting
            proxy = []
            localresult = []
            oracle = get_subtokens(method.name)
            # classname
            proxy.append([self.AST.types[0].name, orederMapping['ClassName']])  # test remove
            # return_type
            if hasattr(method, 'return_type'):
                proxy.append([method.return_type.name, orederMapping['ReturnType']]) if method.return_type else proxy.append(['void', orederMapping['ReturnType']])
                if hasattr(method.return_type, 'dimensions') and method.return_type.dimensions.__len__() != 0:
                    proxy.append(['array', orederMapping['ReturnType']])
            if len(method.parameters) > 0:  # parameters
                for parameter in method.parameters:
                    proxy.append([str(parameter.type).split('name=\'')[1].split('\'')[0], orederMapping['Parameter']])

            # caller
            for caller in self.findCaller(curMethodName):
                if hasattr(caller, 'return_type'):  # return type
                    proxy.append([caller.return_type.name, orederMapping['Caller.ReturnType']]) if caller.return_type else proxy.append(['void', orederMapping['Caller.ReturnType']])
                if len(caller.parameters) > 0:  # parameters
                    for parameter in caller.parameters:
                        proxy.append([str(parameter.type).split('name=\'')[1].split('\'')[0], orederMapping['Caller.Parameter']])


            # callee
            for callee in self.findCallee(curMethodName):
                if str(callee.return_type) != 'None':     # return type
                    proxy.append([callee.return_type.name, orederMapping['Callee.ReturnType']]) if callee.return_type else proxy.append(['void', orederMapping['Callee.ReturnType']])
                if len(callee.parameters) > 0:  # parameters
                    for parameter in callee.parameters:
                        proxy.append([str(parameter.type).split('name=\'')[1].split('\'')[0], orederMapping['Callee.Parameter']])
                if not hasattr(callee, 'tokens') or isinstance(callee.tokens, set):
                    callee.tokens = self.AST.tokens
                callee_tokens = self.getTokenStrem(callee, topk=10)
                for i, x in enumerate(callee_tokens):
                    proxy.append([x.value, orederMapping['Callee.' + x.curType]]) if x.curType != '<UNK>' else \
                        proxy.append([x.value, orederMapping['<UNK>']])

            # token sequence
            method.tokens = self.AST.tokens
            tokens = self.getTokenStrem(method)
            if tokens.__len__() < 3 or tokens.__len__() < oracle.__len__():
                continue
            for i, x in enumerate(tokens):
                proxy.append([x.value, orederMapping[x.curType]])
            if proxy.__len__() <= 0:
                continue

            #before add localresult we need to split it into subtokens and then pad it
            for item in proxy:
                subtokens = get_subtokens(item[0])
                for subtoken in subtokens:
                    if subtoken.isalpha() and len(stripSymbol.sub('', subtoken)) > 1:
                        localresult.append([subtoken, item[1]])
            if localresult.__len__() < 3:
                continue
            # the max_length + 1 is the oracle information
            localresult.append(oracle)
            curMethodRes.append(localresult)
        return curMethodRes


def pad_input(inputData:list, max_length:int):
    if len(inputData) >= max_length:
        return inputData[:max_length]
    else:
        padding_length = max_length - len(inputData)
        unit = [0,0]
        for x in range(padding_length):
            inputData.append(unit)
        return inputData

def saveResult(detailPath, result):
    if not os.path.exists(os.path.dirname(detailPath)):
        os.mkdir(os.path.dirname(detailPath))
    with open(detailPath,'a', encoding='utf-8') as f:
        for x in result:
            f.write(json.dumps(x, separators=(',', ':')) + '\n')

def main(reposPath, outputPath, consideredType, lastRepo=None):
    repos = os.listdir(reposPath)
    if lastRepo is None:
        hasGet = True
    else:
        hasGet = False

    pool = multiprocessing.Pool(processes=process_num)
    for i, repo in enumerate(repos):
        if hasGet or repo == lastRepo:
            hasGet = True
        else:
            continue
        # print('Current repo:', repo)
        if not os.path.isdir(os.path.join(reposPath, repo)):
            continue
        pool.apply_async(core, ((os.path.join(reposPath, repo), outputPath, i, consideredType),))

    pool.close()
    pool.join()

    # args = []
    # for i, repo in enumerate(repos):
    #     if hasGet or repo == lastRepo:
    #         hasGet = True
    #     else:
    #         continue
    #     args.append((os.path.join(reposPath, repo), outputPath, i))
    # with multiprocessing.Pool(processes=process_num) as pool:
    #     list(tqdm(pool.imap(core, iter(args), chunksize=1), total=args.__len__()))
    #
    for i, repo in enumerate(repos):
        if hasGet or repo == lastRepo:
            hasGet = True
        else:
            continue
        core((os.path.join(reposPath, repo), outputPath, i, consideredType))

if __name__ == '__main__':
    consideredType = (javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration)
    ramDiskPath = './ramDisk'
    tarPath = './java-large.tar.gz'


    dataset = "empirical"
    if reposPath is None and os.name == 'posix' and createRamDisk(ramDiskPath, '20480M') == 0 and moveAndDecompress(tarPath, ramDiskPath) == 0:
        reposPath = os.path.join(ramDiskPath, dataset)
    elif reposPath is None:
        reposPath = './' + dataset


    process_num = 1
    lastRepo = None

    reposPath = "./ramDisk/java-large/training"
    outputPath = "./med_test/training.json"
    main(reposPath, outputPath, consideredType, lastRepo)


    

