"""Javascript Unpacker and Simplifier    (c) Jurriaan Bremer, 2013."""
import jsparser
import re


class Base:
    children = []


class String(Base):
    def __init__(self, value):

        # decode unicode parts in the string
        def decode(x):
            # ascii character?
            if int(x.group(1), 16) < 0x80:
                return chr(int(x.group(1), 16))

            # otherwise just keep the original unicode sequence
            return '\\u' + x.group(1)

        self.value = re.sub(r'\\u([0-9a-f]{4})', decode, value)

    def __str__(self):
        return '"%s"' % self.value.replace('"', '\\"')


class Int(Base):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return '%d' % self.value if self.value < 64 else '0x%x' % self.value


class Block(Base):
    children = 'statements',

    def __init__(self, statements):
        self.statements = statements

    def __str__(self):
        return '{\n    %s\n}' % '\n'.join(str(x) for x in self.statements)


class Array(Base):
    children = 'values',

    def __init__(self, typ, values):
        self.typ = typ
        self.values = values

    def __str__(self):
        if self.typ == 'script':
            return '\n'.join(str(x) for x in self.values)
        if self.typ == 'array_init':
            return '[%s]' % ', '.join(str(x) for x in self.values)
        return '(%s)' % ', '.join(str(x) for x in self.values)


class Var(Base):
    children = 'variables',

    def __init__(self, variables):
        self.variables = variables

    def __str__(self):
        return '\n'.join('var %s' % str(x) for x in self.variables)


class Operation(Base):
    children = 'left', 'right'

    def __init__(self, typ, left, right):
        self.typ = typ
        self.left = left
        self.right = right

    def __str__(self):
        if self.right is None:
            return str(self.left) + self.typ
        return '(%s %s %s)' % (str(self.left), self.typ, str(self.right))


class Comparison(Base):
    children = 'left', 'right'

    def __init__(self, typ, left, right):
        self.typ = typ
        self.left = left
        self.right = right

    def __str__(self):
        return '(%s %s %s)' % (str(self.left), self.typ, str(self.right))


class Conditional(Base):
    children = 'condition', 'then', 'else_'

    def __init__(self, condition, then, else_):
        self.condition = condition
        self.then = then
        self.else_ = else_

    def __str__(self):
        ret = 'if%s\n%s\n' % (str(self.condition), str(self.then))
        if self.else_:
            ret += '\nelse\n%s\n' % str(self.else_)
        return ret


class Call(Base):
    children = 'function', 'params'

    def __init__(self, function, params):
        self.function = function
        self.params = params

    def __str__(self):
        return str(self.function) + str(self.params)


class Function(Base):
    children = 'function',

    def __init__(self, function, params):
        self.function = function
        self.params = params

    def __str__(self):
        return 'function(%s) {\n    %s\n}' % (', '.join(self.params),
                                              str(self.function))


class Identifier(Base):
    children = 'initializer',

    def __init__(self, name, initializer=None):
        self.name = name
        self.initializer = initializer

    def __str__(self):
        if not self.initializer is None:
            return '%s = %s' % (self.name, str(self.initializer))
        return self.name


class New(Base):
    children = 'identifier', 'args'

    def __init__(self, identifier, args=[]):
        self.identifier = identifier
        self.args = args

    def __str__(self):
        return 'new %s(%s)' % (str(self.identifier),
                               ', '.join(str(x) for x in self.args))


class For(Base):
    children = 'setup', 'condition', 'update', 'body'

    def __init__(self, setup, condition, update, body):
        self.setup = setup
        self.condition = condition
        self.update = update
        self.body = body

    def __str__(self):
        return 'for (%s;%s;%s)\n%s\n' % (str(self.setup),
                                         str(self.condition),
                                         str(self.update),
                                         str(self.body))


class Assign(Base):
    children = 'left', 'right'

    def __init__(self, typ, left, right):
        self.typ = typ
        self.left = left
        self.right = right

    def __str__(self):
        return '%s %s %s' % (str(self.left),
                             self.typ if self.typ == '=' else self.typ + '=',
                             str(self.right))


class Dot(Base):
    children = 'left', 'right'

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return '%s.%s' % (str(self.left), str(self.right))


class Index(Base):
    children = 'array', 'index'

    def __init__(self, array, index):
        self.array = array
        self.index = index

    def __str__(self):
        return '%s[%s]' % (str(self.array), str(self.index))


class Typeof(Base):
    children = 'value',

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'typeof%s' % str(self.value)


class Constant(Base):
    def __init__(self, typ):
        self.typ = typ

    def __str__(self):
        return self.typ


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
    'group': _Translator('group', parser='array'),
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


class Simplifier:
    """Simplifies trees of Javascript objects."""

    def __init__(self, root):
        self.root = root
        self.simplified = False

    def __str__(self):
        return str(self.simplify())

    def simplify(self):
        if self.simplified:
            return self.root

        def walk(node, simplifier):
            self.count += 1
            for name, x in ((name, getattr(node, name))
                            for name in node.children):
                if isinstance(x, Base):
                    x = walk(simplifier(x), simplifier)
                    setattr(node, name, simplifier(x))
                elif x:
                    x = [simplifier(y) for y in x]
                    x = [walk(y, simplifier) for y in x]
                    setattr(node, name, [simplifier(y) for y in x])

            return node

        length, self.count = -1, 0
        while length != self.count:
            length, self.count = self.count, 0
            for simplifier in self.simplifiers:
                self.root = walk(self.root, getattr(self, simplifier))

        return self.root

    simplifiers = [
        '_concat_strings',
        '_empty_group',
        '_from_char_code',
        '_hardcoded_obj_calls',
        '_index_string',
        '_parse_int',
    ]

    def _concat_strings(self, node):
        if isinstance(node, Operation) and node.typ == '+' and \
                isinstance(node.left, String) and \
                isinstance(node.right, String):
            return String(node.left.value + node.right.value)
        return node

    def _empty_group(self, node):
        if isinstance(node, Array) and node.typ == 'group' and \
                len(node.values) == 1 and \
                isinstance(node.values[0], (Int, String)):
            return node.values[0]
        return node

    def _from_char_code(self, node):
        if isinstance(node, Call) and isinstance(node.function, Dot) and \
                str(node.function) == 'String.fromCharCode' and \
                isinstance(node.params.values[0], Int):
            return String(chr(node.params.values[0].value))
        return node

    def _hardcoded_obj_calls(self, node):
        if not isinstance(node, Call):
            return node

        tbl = {
            (String, 'toLowerCase'): lambda x: String(x.lower()),
            (String, 'toUpperCase'): lambda x: String(x.upper()),
            (String, 'toString'): lambda x: String(x),
            (Int, 'toString'): lambda x: String(str(x)),
        }

        fn = node.function
        if isinstance(fn, Dot) and isinstance(fn.left, (Int, String)) and \
                isinstance(fn.right, Identifier) and \
                (fn.left.__class__, fn.right.name) in tbl:
            return tbl[fn.left.__class__, fn.right.name](fn.left.value)
        return node

    def _index_string(self, node):
        if not isinstance(node, Index):
            return node

        if isinstance(node.array, Identifier) and \
                isinstance(node.index, String):
            return Dot(node.array, Identifier(node.index.value))
        return node

    def _parse_int(self, node):
        if not isinstance(node, Call):
            return node

        params = node.params.values
        if isinstance(node.function, Identifier) and len(params) == 2 and \
                isinstance(params[0], String) and isinstance(params[1], Int):
            return Int(int(params[0].value, params[1].value))
        return node

if __name__ == '__main__':
    import jsbeautifier
    import sys
    obj = parse(open(sys.argv[1], 'rb').read())

    print jsbeautifier.beautify(str(Simplifier(obj)))
