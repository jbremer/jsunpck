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


class Array(Base):
    def __init__(self, typ, values):
        self.typ = typ
        self.values = values


class Operation(Base):
    def __init__(self, typ, value1, value2):
        self.typ = typ
        self.value1 = value1
        self.value2 = value2


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
    def __init__(self, identifier):
        self.identifier = identifier


class For(Base):
    def __init__(self, setup, condition, update, body):
        self.setup = setup
        self.condition = condition
        self.update = update
        self.body = body


class Assign(Base):
    def __init__(self, left, right):
        self.left = left
        self.right = right


class Index(Base):
    def __init__(self, array, index):
        self.array = array
        self.index = index


class Constant(Base):
    def __init__(self, typ):
        self.typ = typ


class _Base:
    def __init__(self, typ, *args, **kwargs):
        self.typ = typ
        self.args = args
        self.kwargs = kwargs


class _String(_Base):
    def parse(self, node):
        return String(node.value)


class _Int(_Base):
    def parse(self, node):
        return Int(node.value)


class _Array(_Base):
    def parse(self, node):
        return Array(self.typ, [_parse(node[x]) for x in xrange(len(node))])


class _Container(_Base):
    def parse(self, node):
        if self.typ == 'semicolon':
            return _parse(node.expression)
        if self.typ == 'var':
            return Array('assign',
                         [_parse(node[x]) for x in xrange(node.length)])
        raise Exception(self.typ)


class _Operation(_Base):
    def parse(self, node):
        if len(node) == 1:
            return Operation(self.typ, node[0], None)
        return Operation(self.typ, node[0], node[1])


class _Comparison(_Base):
    def parse(self, node):
        return Comparison(node.value, node[0], node[1])


class _Conditional(_Base):
    def parse(self, node):
        then = _parse(node.thenPart) if node.thenPart else None
        else_ = _parse(node.elsePart) if node.elsePart else None
        return Conditional(condition=_parse(node.condition),
                           then=then,
                           else_=else_)


class _Call(_Base):
    def parse(self, node):
        return Call(_parse(node[0]), _parse(node[1]))


class _Function(_Base):
    def parse(self, node):
        return Function(_parse(node.body), node.params)


class _Identifier(_Base):
    def parse(self, node):
        if hasattr(node, 'initializer'):
            return Identifier(node.name, _parse(node.initializer))
        return Identifier(node.value, None)


class _New(_Base):
    def parse(self, node):
        return New(_parse(node[0]))


class _For(_Base):
    def parse(self, node):
        return For(setup=_parse(node.setup),
                   condition=_parse(node.condition),
                   update=_parse(node.update),
                   body=_parse(node.body))


class _Assign(_Base):
    def parse(self, node):
        return Assign(left=node[0], right=node[1])


class _Index(_Base):
    def parse(self, node):
        return Index(node[0], node[1])


class _Constant(_Base):
    def parse(self, node):
        return Constant(node.value)

# rules to extract all relevant fields from the javascript tokens
rules = {
    'script': _Array('script'),
    'group': _Array('group'),
    'var': _Array('var'),
    'list': _Array('list'),
    'array_init': _Array('array_init'),
    'block': _Array('block'),

    'semicolon': _Container('semicolon'),
    'identifier': _Identifier('identifier'),

    'string': _String(''),
    'number': _Int(''),

    'plus': _Operation('+', _Base, _Base),
    'minus': _Operation('-', _Base, _Base),
    'mod': _Operation('%', _Base, _Base),
    'bitwise_xor': _Operation('^', _Base, _Base),
    'and': _Operation('&&', _Base, _Base),
    'increment': _Operation('++', _Base, None, postfix='postfix'),

    'lt': _Comparison('<', _Base, _Base),
    'gt': _Comparison('>', _Base, _Base),
    'eq': _Comparison('==', _Base, _Base),
    'ne': _Comparison('!=', _Base, _Base),

    'if': _Conditional('if'),

    'call': _Call('call'),
    'function': _Function('function'),
    'new': _New('new'),
    'for': _For('for'),
    'assign': _Assign('assign'),
    'index': _Index('index'),

    'true': _Constant('true'),
    'false': _Constant('false'),
}


def _parse(node):
    """Really parse a (sub-)node."""
    token = jsparser.tokenstr(node.type_).lower()
    if token not in rules:
        print node
        raise Exception('%s not supported' % token)

    return rules[token].parse(node)


def parse(source):
    """Parses javascript and translates it into our object model."""
    return _parse(jsparser.parse(source))

if __name__ == '__main__':
    import sys
    print parse(open(sys.argv[1], 'rb').read())
