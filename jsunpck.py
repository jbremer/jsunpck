"""Javascript Unpacker and Simplifier    (c) Jurriaan Bremer, 2013."""
import jsparser


class Base:
    pass


class String(Base):
    def __init__(self, value):
        self.value = value


class Int(Base):
    def __init__(self, value):
        self.value = value


class Block(Base):
    def __init__(self, statements):
        self.statements = statements


class Array(Base):
    def __init__(self, typ, values):
        self.typ = typ
        self.values = values


class Var(Base):
    def __init__(self, variables):
        self.variables = variables


class Operation(Base):
    def __init__(self, typ, left, right):
        self.typ = typ
        self.left = left
        self.right = right


class Comparison(Base):
    def __init__(self, typ, left, right):
        self.typ = typ
        self.left = left
        self.right = right


class Conditional(Base):
    def __init__(self, condition, then, else_):
        self.condition = condition
        self.then = then
        self.else_ = else_


class Call(Base):
    def __init__(self, function, params):
        self.function = function
        self.params = params


class Function(Base):
    def __init__(self, function, params):
        self.function = function
        self.params = params


class Identifier(Base):
    def __init__(self, name, initializer):
        self.name = name
        self.initializer = initializer


class New(Base):
    def __init__(self, identifier, args=[]):
        self.identifier = identifier
        self.args = args


class For(Base):
    def __init__(self, setup, condition, update, body):
        self.setup = setup
        self.condition = condition
        self.update = update
        self.body = body


class Assign(Base):
    def __init__(self, typ, left, right):
        self.typ = typ
        self.left = left
        self.right = right


class Dot(Base):
    def __init__(self, left, right):
        self.left = left
        self.right = right


class Index(Base):
    def __init__(self, array, index):
        self.array = array
        self.index = index


class Typeof(Base):
    def __init__(self, value):
        self.value = value


class Constant(Base):
    def __init__(self, typ):
        self.typ = typ


class _Translator:
    def __init__(self, typ=None, parser=None, **kwargs):
        self.typ = typ
        self.parser = parser
        self.__dict__.update(kwargs)

    def parse(self, token, node):
        return getattr(self, '_' + (self.parser or token))(node)

    def _string(self, node):
        return String(node.value)

    def _number(self, node):
        return Int(node.value)

    def _block(self, node):
        return Block([_parse(node[x]) for x in xrange(len(node))])

    def _array(self, node):
        return Array(self.typ, [_parse(node[x]) for x in xrange(len(node))])

    def _var(self, node):
        return Var([_parse(node[x]) for x in xrange(len(node))])

    def _container(self, node):
        if self.typ == 'semicolon':
            return _parse(node.expression)
        raise Exception(self.typ)

    def _operation(self, node):
        if len(node) == 1:
            return Operation(self.typ, _parse(node[0]), None)
        return Operation(self.typ, _parse(node[0]), _parse(node[1]))

    def _comparison(self, node):
        return Comparison(node.value, _parse(node[0]), _parse(node[1]))

    def _conditional(self, node):
        then = _parse(node.thenPart) if node.thenPart else None
        else_ = _parse(node.elsePart) if node.elsePart else None
        return Conditional(condition=_parse(node.condition),
                           then=then,
                           else_=else_)

    def _call(self, node):
        return Call(_parse(node[0]), _parse(node[1]))

    def _function(self, node):
        return Function(_parse(node.body), node.params)

    def _identifier(self, node):
        if hasattr(node, 'initializer'):
            return Identifier(node.name, _parse(node.initializer))
        return Identifier(node.value, None)

    def _new(self, node):
        return New(_parse(node[0]))

    def _new_with_args(self, node):
        return New(_parse(node[0]), [_parse(x) for x in node[1]])

    def _for(self, node):
        return For(setup=_parse(node.setup),
                   condition=_parse(node.condition),
                   update=_parse(node.update),
                   body=_parse(node.body))

    def _assign(self, node):
        return Assign(node.value, _parse(node[0]), _parse(node[1]))

    def _index(self, node):
        return Index(_parse(node[0]), _parse(node[1]))

    def _dot(self, node):
        return Dot(_parse(node[0]), _parse(node[1]))

    def _typeof(self, node):
        return Typeof(_parse(node[0]))

    def _constant(self, node):
        return Constant(node.value)

# rules to extract all relevant fields from the javascript tokens
rules = {
    'script': _Translator('script', parser='array'),
    'group': _Translator(parser='array'),
    'var': _Translator(),
    'list': _Translator(parser='array'),
    'array_init': _Translator('array_init', parser='array'),
    'block': _Translator(),

    'semicolon': _Translator('semicolon', parser='container'),
    'identifier': _Translator(),

    'string': _Translator(),
    'number': _Translator(),

    'plus': _Translator('+', parser='operation'),
    'minus': _Translator('-', parser='operation'),
    'mod': _Translator('%', parser='operation'),
    'div': _Translator('/', parser='operation'),
    'bitwise_xor': _Translator('^', parser='operation'),
    'and': _Translator('&&', parser='operation'),
    'increment': _Translator('++', parser='operation', postfix='postfix'),

    'lt': _Translator('<', parser='comparison'),
    'gt': _Translator('>', parser='comparison'),
    'eq': _Translator('==', parser='comparison'),
    'ne': _Translator('!=', parser='comparison'),

    'if': _Translator(parser='conditional'),

    'call': _Translator(),
    'function': _Translator(),
    'new': _Translator(),
    'new_with_args': _Translator(),
    'for': _Translator(),
    'assign': _Translator(),
    'index': _Translator(),
    'dot': _Translator(),
    'typeof': _Translator(),

    'true': _Translator(parser='constant'),
    'false': _Translator(parser='constant'),
}


def _parse(node):
    """Really parse a (sub-)node."""
    token = jsparser.tokenstr(node.type_).lower()
    if token not in rules:
        print node
        raise Exception('%s not supported' % token)

    return rules[token].parse(token, node)


def parse(source):
    """Parses javascript and translates it into our object model."""
    return _parse(jsparser.parse(source))

if __name__ == '__main__':
    import sys
    print parse(open(sys.argv[1], 'rb').read())
