"""Javascript Unpacker and Simplifier    (c) Jurriaan Bremer, 2013."""
import jsparser
import re


def base_n(num, b, chrs='0123456789abcdefghijklmnopqrstuvwxyz'):
    return '0' if not num else base_n(num // b, b).lstrip('0') + chrs[num % b]


class Base:
    children = []

    def __cmp__(self, other):
        if other.__class__ == Base:
            return -1

        if self.__class__ != other.__class__:
            return 1

        for x in self.children:
            if cmp(getattr(self, x), getattr(other, x)) > 0:
                return 1

        return 0


class String(Base):
    def __init__(self, value=None):

        # decode unicode parts in the string
        def decode(x):
            # ascii character?
            if int(x.group(1), 16) < 0x80:
                return chr(int(x.group(1), 16))

            # otherwise just keep the original unicode sequence
            return '\\u' + x.group(1)

        self.value = None
        if not value is None:
            self.value = re.sub(r'\\u([0-9a-f]{4})', decode, value)

    def __str__(self):
        return '"%s"' % self.value.replace('"', '\\"')

    def __cmp__(self, other):
        base = Base.__cmp__(self, other)

        if not base and hasattr(other, 'value') and other.value is None:
            return 0

        return base or self.value != other.value


class Int(Base):
    def __init__(self, value=None):
        self.value = value

    def __str__(self):
        return '%d' % self.value if self.value < 64 else '0x%x' % self.value

    def __cmp__(self, other):
        base = Base.__cmp__(self, other)

        if not base and hasattr(other, 'value') and other.value is None:
            return 0

        return base or self.value != other.value


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

    def __cmp__(self, other):
        return Base.__cmp__(self, other) or self.typ != other.typ


class Return(Base):
    children = 'value',

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'return ' + str(self.value)


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
            return self.typ + str(self.left)
        return '(%s %s %s)' % (str(self.left), self.typ, str(self.right))

    def __cmp__(self, other):
        return Base.__cmp__(self, other) or self.typ != other.typ


class Comparison(Base):
    children = 'left', 'right'

    def __init__(self, typ, left, right):
        self.typ = typ
        self.left = left
        self.right = right

    def __str__(self):
        return '(%s %s %s)' % (str(self.left), self.typ, str(self.right))

    def __cmp__(self, other):
        return Base.__cmp__(self, other) or self.typ != other.typ


class Conditional(Base):
    children = 'condition', 'then', 'else_'

    def __init__(self, condition, then, else_):
        self.condition = condition
        self.then = then
        self.else_ = else_

    def __str__(self):
        if isinstance(self.condition, (Int, String, Identifier, Constant)):
            ret = 'if(%s)' % str(self.condition)
        else:
            ret = 'if %s' % str(self.condition)

        ret += '\n%s\n' % str(self.then)
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

    def __cmp__(self, other):
        return Base.__cmp__(self, other) or self.params != other.params


class Identifier(Base):
    children = 'initializer',

    def __init__(self, name=None, initializer=None):
        self.name = name
        self.initializer = initializer

    def __str__(self):
        if not self.initializer is None:
            return '%s = %s' % (self.name, str(self.initializer))
        return self.name

    def __cmp__(self, other):
        return Base.__cmp__(self, other) or \
            (self.name != other.name and not other.name is None)


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

    def __cmp__(self, other):
        return Base.__cmp__(self, other) or self.typ != other.typ


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
        return 'typeof(%s)' % str(self.value)


class Constant(Base):
    def __init__(self, typ):
        self.typ = typ

    def __str__(self):
        return self.typ

    def __cmp__(self, other):
        return Base.__cmp__(self, other) or self.typ != other.typ


class Try(Base):
    children = 'try_block', 'catch_clauses'

    def __init__(self, try_block, catch_clauses):
        self.try_block = try_block
        self.catch_clauses = catch_clauses

    def __str__(self):
        return 'try %s %s' % (str(self.try_block),
                              '\n'.join(str(x) for x in self.catch_clauses))


class Catch(Base):
    children = 'block',

    def __init__(self, var_name, block):
        self.var_name = var_name
        self.block = block

    def __str__(self):
        return 'catch (%s) %s' % (self.var_name, str(self.block))


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

    def _return(self, node):
        return Return(_parse(node.value))

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

    def _try(self, node):
        return Try(_parse(node.tryBlock),
                   [_parse(x) for x in node.catchClauses])

    def _catch(self, node):
        return Catch(node.varName, _parse(node.block))

# rules to extract all relevant fields from the javascript tokens
rules = {
    'script': _Translator('script', parser='array'),
    'group': _Translator('group', parser='array'),
    'var': _Translator(),
    'list': _Translator(parser='array'),
    'array_init': _Translator('array_init', parser='array'),
    'block': _Translator(),
    'return': _Translator(),

    'semicolon': _Translator('semicolon', parser='container'),
    'identifier': _Translator(),

    'string': _Translator(),
    'number': _Translator(),

    'plus': _Translator('+', parser='operation'),
    'minus': _Translator('-', parser='operation'),
    'mul': _Translator('*', parser='operation'),
    'mod': _Translator('%', parser='operation'),
    'div': _Translator('/', parser='operation'),
    'unary_minus': _Translator('-', parser='operation'),
    'bitwise_not': _Translator('~', parser='operation'),
    'bitwise_xor': _Translator('^', parser='operation'),
    'bitwise_or': _Translator('|', parser='operation'),
    'lsh': _Translator('<<', parser='operation'),
    'rsh': _Translator('>>', parser='operation'),
    'and': _Translator('&&', parser='operation'),
    'or': _Translator('||', parser='operation'),
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

    'try': _Translator(),
    'catch': _Translator(),
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

        self.variables = {}

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
                    x = walk(simplifier(x) or x, simplifier)
                    setattr(node, name, simplifier(x) or x)
                elif x:
                    x = [simplifier(y) or y for y in x]
                    x = [walk(y, simplifier) for y in x]
                    setattr(node, name, [simplifier(y) or y for y in x])

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
        '_hardcoded_obj_calls',
        '_index_string',
        '_parse_int',
        '_rename_variables',
        '_subtract_itself',
        '_const_arithmetic',
        '_const_str_length',
        '_single_return_value',
        '_const_comparison',
    ]

    def _concat_strings(self, node):
        if node == Operation('+', String(), String()):
            return String(node.left.value + node.right.value)

    def _empty_group(self, node):
        if node in (Array('group', [Int()]),
                    Array('group', [String()]),
                    Array('group', [Identifier()])):
            return node.values[0]

    def _hardcoded_obj_calls(self, node):
        if not isinstance(node, Call):
            return node

        fn, params = node.function, node.params.values

        # Int.toString() or Int.toString(base)
        if fn == Dot(Int(), Identifier('toString')) and \
                params in ([], [Int()]):
            base = 10 if not len(params) else params[0].value
            return String(base_n(int(fn.left.value), base))

        # String.fromCharCode(Int(), Int(), Int(), ...)
        if fn == Dot(Identifier('String'), Identifier('fromCharCode')) and \
                not [x for x in params if not isinstance(x, Int)]:
            return String(''.join(chr(x.value) for x in params))

        tbl = {
            'toLowerCase': lambda x: x.lower(),
            'toUpperCase': lambda x: x.upper(),
            'toString': lambda x: x,
        }

        if fn == Dot(String(), Identifier()) and fn.right.name in tbl:
            return String(tbl[fn.right.name](fn.left.value))

    def _index_string(self, node):
        if node == Index(Base(), String()):
            if not node.index.value.isdigit():
                return Dot(node.array, Identifier(node.index.value))
            else:
                return Index(node.array, Int(int(node.index.value)))
        if node == Index(Base(), Call(Dot(Base(), Identifier('toString')),
                                      Array(None, []))):
            return Index(node.array, node.index.function.left)

    def _parse_int(self, node):
        if node == Call(Identifier('parseInt'),
                        Array(None, [String(), Int()])):
            return Int(int(node.params.values[0].value,
                           node.params.values[1].value))

    def _rename_variables(self, node):
        if isinstance(node, Var):
            for var in node.variables:
                if var.name[:4] != 'var_' and not var.name in self.variables:
                    self.variables[var.name] = 'var_%d' % len(self.variables)

        if isinstance(node, Function):
            for idx, var in enumerate(node.params):
                if var[:6] != 'param_' and not var in self.variables:
                    self.variables[var] = 'param_%d' % len(self.variables)
                    node.params[idx] = self.variables[var]

        if isinstance(node, Identifier) and node.name in self.variables:
            return Identifier(self.variables[node.name], node.initializer)

    def _subtract_itself(self, node):
        if node == Operation('-', Identifier(), Identifier()) and \
                node.left.name == node.right.name:
            return Int(0)

    def _const_arithmetic(self, node):
        tbl = {
            '-': lambda x: -x,
            '~': lambda x: ~x,
        }
        tbl2 = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '%': lambda x, y: x % y,
            '>>': lambda x, y: x >> y,
            '<<': lambda x, y: x << y,
            '^': lambda x, y: x ^ y,
            '|': lambda x, y: x | y,
            '&&': lambda x, y: y if x else 0,
            '||': lambda x, y: x if x else y,
        }
        if Base.__cmp__(node, Operation(None, Int(), Int())) == 0:
            if node.right is None:
                return Int(tbl[node.typ](node.left.value))

            if node.typ in ('/', '%') and not node.right.value:
                return

            return Int(tbl2[node.typ](node.left.value, node.right.value))

    def _const_str_length(self, node):
        if node == Dot(String(), Identifier('length')):
            return Int(len(node.left.value))

    def _single_return_value(self, node):
        # function with no parameters and only a return statement
        if node == Function(Array('script', [Return(Base())]), []):
            return node.function.values[0].value

    def _const_comparison(self, node):
        tbl = {
            '<': lambda x, y: x < y,
            '>': lambda x, y: x > y,
            '==': lambda x, y: x == y,
            '!=': lambda x, y: x != y,
        }

        if Base.__cmp__(node, Comparison(None, Int(), Int())) == 0 and \
                node.typ in tbl:
            val = tbl[node.typ](node.left.value, node.right.value)
            return Constant('true' if val else 'false')

if __name__ == '__main__':
    import jsbeautifier
    import sys
    obj = parse(open(sys.argv[1], 'rb').read())

    print jsbeautifier.beautify(str(Simplifier(obj)))
