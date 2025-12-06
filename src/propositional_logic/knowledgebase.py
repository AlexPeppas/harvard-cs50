class Symbol:
    def __init__(self):
        self.args_no = 0

    def ser(self):
        if (self.args_no > 1):
            return f'({self.content})'
        else:
            return self.content

class And(Symbol):
    def __init__(self, args):
        self.args_no = len(args)
        self.content = " ^ ".join(args)

class Or(Symbol):
    def __init__(self, args):
        self.args_no = len(args)
        self.content = " v ".join(args)

class Not(Symbol):
    def __init__(self, arg):
        self.args_no = 1
        self.content = " ¬ " + arg

class KnowledgeBaseBuilder:
    
    def __init__(self):
        self.expressions = set()
    
    def create(self):
        return self
    
    def add(self, expression):
        self.expressions.add(expression)
        return self
    
    def format(self):
        for exp in self.expressions:
            print(f'{exp}\n')


kb = KnowledgeBaseBuilder().create()
kb.add(And([Or(["alex", "hawking", "lane"]).ser(), Or(["house", "outdoors"]).ser()]).ser())
kb.add(Not("hawking").ser())
kb.format()