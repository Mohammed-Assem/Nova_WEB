from highlight_utils import *
import string

DIGITS = '0123456789'
LETTERS = string.ascii_letters
CHARS = LETTERS + DIGITS

class Cursor:
    def __init__(self, idx, line, col, file_name, file_text):
        self.idx = idx
        self.line = line
        self.col = col
        self.file_name = file_name
        self.file_text = file_text

    def move(self, current=None):
        self.idx += 1
        self.col += 1
        if current == '\n':
            self.line += 1
            self.col = 0
        return self

    def clone(self):
        return Cursor(self.idx, self.line, self.col, self.file_name, self.file_text)

class NovaError:
    def __init__(self, start, end, name, msg):
        self.start = start
        self.end = end
        self.name = name
        self.msg = msg

    def describe(self):
        result = f"{self.name}: {self.msg}\n"
        if self.start and self.end:
            result += f"File {self.start.file_name}, line {self.start.line + 1}\n"
            result += "\n" + point_to_error(self.start.file_text, self.start, self.end)
        return result


class SyntaxError(NovaError):
    def __init__(self, start, end, msg="Syntax error"):
        super().__init__(start, end, "SyntaxError", msg)

class NovaRuntimeError(NovaError, Exception):
    def __init__(self, start, end, msg, context):
        super().__init__(start, end, "RuntimeError", msg)
        self.context = context

TT_INT = "INT"
TT_FLOAT = "FLOAT"
TT_STRING = "STRING"
TT_IDENTIFIER = "ID"
TT_KEYWORD = "KEY"
TT_PLUS = "PLUS"
TT_MINUS = "MINUS"
TT_MUL = "MUL"
TT_DIV = "DIV"
TT_EQ = "EQ"
TT_EE = "EE"
TT_NE = "NE"
TT_LT = "LT"
TT_GT = "GT"
TT_LTE = "LTE"
TT_GTE = "GTE"
TT_LPAREN = "LPAREN"
TT_RPAREN = "RPAREN"
TT_ARROW = "ARROW"
TT_COMMA = "COMMA"
TT_EOF = "EOF"

KEYWORDS = [
    "Nova", "FUNC", "IF", "THEN", "ELSE", "WHILE", "DO", "FOR", "TO", "STEP",
    "TRUE", "FALSE", "AND", "OR", "NOT", "RETURN"
]

class Token:
    def __init__(self, type_, value=None, start=None, end=None):
        self.type = type_
        self.value = value
        self.start = start.clone() if start else None
        self.end = end.clone() if end else None

    def __repr__(self):
        return f'{self.type}:{self.value}' if self.value else self.type

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

class NovaLexer:
    def __init__(self, file_name, text):
        self.text = text
        self.file_name = file_name
        self.cursor = Cursor(-1, 0, -1, file_name, text)
        self.char = None
        self.advance()

    def advance(self):
        self.cursor.move(self.char)
        self.char = self.text[self.cursor.idx] if self.cursor.idx < len(self.text) else None

    def tokenize(self):
        tokens = []
        while self.char:
            if self.char in ' \t\r\n':
                self.advance()
            elif self.char in DIGITS:
                tokens.append(self.make_number())
            elif self.char in LETTERS:
                tokens.append(self.make_identifier())
            elif self.char == '"':
                tokens.append(self.make_string())
            elif self.char == '+':
                tokens.append(Token(TT_PLUS, start=self.cursor))
                self.advance()
            elif self.char == '-':
                start = self.cursor.clone()
                self.advance()
                if self.char == '>':
                    self.advance()
                    tokens.append(Token(TT_ARROW, start=start, end=self.cursor))
                else:
                    tokens.append(Token(TT_MINUS, start=start))
            elif self.char == '*':
                tokens.append(Token(TT_MUL, start=self.cursor))
                self.advance()
            elif self.char == '/':
                tokens.append(Token(TT_DIV, start=self.cursor))
                self.advance()
            elif self.char == '=':
                start = self.cursor.clone()
                self.advance()
                if self.char == '=':
                    self.advance()
                    tokens.append(Token(TT_EE, start=start, end=self.cursor))
                else:
                    tokens.append(Token(TT_EQ, start=start))
            elif self.char == '#':
                self.skip_comment()
            elif self.char == '!':
                start = self.cursor.clone()
                self.advance()
                if self.char == '=':
                    self.advance()
                    tokens.append(Token(TT_NE, start=start, end=self.cursor))
                else:
                    return [], SyntaxError(start, self.cursor, "Expected '=' after '!'")
            elif self.char == '<':
                start = self.cursor.clone()
                self.advance()
                if self.char == '=':
                    self.advance()
                    tokens.append(Token(TT_LTE, start=start, end=self.cursor))
                else:
                    tokens.append(Token(TT_LT, start=start))
            elif self.char == '>':
                start = self.cursor.clone()
                self.advance()
                if self.char == '=':
                    self.advance()
                    tokens.append(Token(TT_GTE, start=start, end=self.cursor))
                else:
                    tokens.append(Token(TT_GT, start=start))
            elif self.char == '(':
                tokens.append(Token(TT_LPAREN, start=self.cursor))
                self.advance()
            elif self.char == ')':
                tokens.append(Token(TT_RPAREN, start=self.cursor))
                self.advance()
            elif self.char == ',':
                tokens.append(Token(TT_COMMA, start=self.cursor))
                self.advance()
            else:
                start = self.cursor.clone()
                char = self.char
                self.advance()
                return [], SyntaxError(start, self.cursor, f"Illegal character '{char}'")
        tokens.append(Token(TT_EOF, start=self.cursor))
        return tokens, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        start = self.cursor.clone()

        while self.char and self.char in DIGITS + '.':
            if self.char == '.':
                if dot_count == 1:
                    break
                dot_count += 1
            num_str += self.char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), start, self.cursor)
        else:
            return Token(TT_FLOAT, float(num_str), start, self.cursor)
    def skip_comment(self):
        self.advance()
        while self.char and self.char != '\n':
            self.advance()
    def make_string(self):
        string = ''
        start = self.cursor.clone()
        escape = False
        self.advance()

        escapes = {'n': '\n', 't': '\t'}

        while self.char and (self.char != '"' or escape):
            if escape:
                string += escapes.get(self.char, self.char)
                escape = False
            else:
                if self.char == '\\':
                    escape = True
                else:
                    string += self.char
            self.advance()

        self.advance()
        return Token(TT_STRING, string, start, self.cursor)

    def make_identifier(self):
        id_str = ''
        start = self.cursor.clone()
        while self.char and self.char in CHARS + '_':
            id_str += self.char
            self.advance()

        token_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
        return Token(token_type, id_str, start, self.cursor)
    
class NumberNode:
    def __init__(self, tok):
        self.tok = tok

class StringNode:
    def __init__(self, tok):
        self.tok = tok

class BoolNode:
    def __init__(self, value_tok):
        self.tok = value_tok

class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok

class VarAssignNode:
    def __init__(self, var_name_tok, value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node

class BinOpNode:
    def __init__(self, left, op_tok, right):
        self.left = left
        self.op_tok = op_tok
        self.right = right

class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

class IfNode:
    def __init__(self, condition_node, true_node, false_node):
        self.condition_node = condition_node
        self.true_node = true_node
        self.false_node = false_node

class WhileNode:
    def __init__(self, condition_node, body_node):
        self.condition_node = condition_node
        self.body_node = body_node

class ForNode:
    def __init__(self, var_name_tok, start_node, end_node, step_node, body_node):
        self.var_name_tok = var_name_tok
        self.start_node = start_node
        self.end_node = end_node
        self.step_node = step_node
        self.body_node = body_node

class FuncDefNode:
    def __init__(self, name_tok, arg_name_toks, body_node):
        self.name_tok = name_tok
        self.arg_name_toks = arg_name_toks
        self.body_node = body_node
        
class ReturnNode:
    def __init__(self, value_node):
        self.value_node = value_node

class CallNode:
    def __init__(self, node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes

class ProgramNode:
    def __init__(self, statements):
        self.statements = statements

class NovaParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.idx = -1
        self.advance()

    def advance(self):
        self.idx += 1
        if self.idx < len(self.tokens):
            self.current = self.tokens[self.idx]
        return self.current

    def parse(self):
        statements = []

        while self.current.type != TT_EOF:
            if self.current.type == TT_EOF:
                break

            expr = self.expr()
            statements.append(expr)

            if self.current.type == TT_EOF:
                break
            if self.current.type == TT_EOF or self.current.value == '\n':
                self.advance()

        return ProgramNode(statements)


    def expr(self):
        if self.current.matches(TT_KEYWORD, 'Nova'):
            self.advance()
            if self.current.type != TT_IDENTIFIER:
                raise Exception("Expected identifier after Nova")
            var_name = self.current
            self.advance()
            if self.current.type != TT_EQ:
                raise Exception("Expected '=' after identifier")
            self.advance()
            expr = self.expr()
            return VarAssignNode(var_name, expr)

        node = self.comp_expr()

        if self.current.type == TT_EQ:
            if not isinstance(node, VarAccessNode):
                raise Exception("Invalid assignment target")
            self.advance()
            value = self.expr()
            return VarAssignNode(node.var_name_tok, value)

        return node


    def comp_expr(self):
        if self.current.matches(TT_KEYWORD, 'NOT'):
            op = self.current
            self.advance()
            node = self.comp_expr()
            return UnaryOpNode(op, node)

        node = self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE))
        return node

    def arith_expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def term(self):
        return self.bin_op(self.factor, (TT_MUL, TT_DIV))

    def factor(self):
        tok = self.current

        if tok.type in (TT_PLUS, TT_MINUS):
            self.advance()
            node = self.factor()
            return UnaryOpNode(tok, node)

        elif tok.type in (TT_INT, TT_FLOAT):
            self.advance()
            return NumberNode(tok)

        elif tok.type == TT_STRING:
            self.advance()
            return StringNode(tok)

        elif tok.matches(TT_KEYWORD, 'TRUE') or tok.matches(TT_KEYWORD, 'FALSE'):
            self.advance()
            return BoolNode(tok)

        elif tok.type == TT_IDENTIFIER:
            self.advance()
            if self.current.type == TT_LPAREN:
                self.advance()
                arg_nodes = []
                if self.current.type != TT_RPAREN:
                    arg_nodes.append(self.expr())
                    while self.current.type == TT_COMMA:
                        self.advance()
                        arg_nodes.append(self.expr())
                if self.current.type != TT_RPAREN:
                    raise Exception("Expected ')'")
                self.advance()
                return CallNode(VarAccessNode(tok), arg_nodes)
            else:
                return VarAccessNode(tok)


        elif tok.type == TT_LPAREN:
            self.advance()
            expr = self.expr()
            if self.current.type == TT_RPAREN:
                self.advance()
                return expr
            else:
                raise Exception("Expected ')'")

        elif tok.matches(TT_KEYWORD, 'IF'):
            return self.if_expr()

        elif tok.matches(TT_KEYWORD, 'WHILE'):
            return self.while_expr()

        elif tok.matches(TT_KEYWORD, 'FOR'):
            return self.for_expr()

        elif tok.matches(TT_KEYWORD, 'FUNC'):
            return self.func_def()

        raise Exception("Invalid syntax")

    def if_expr(self):
        self.advance()
        condition = self.expr()
        if not self.current.matches(TT_KEYWORD, 'THEN'):
            raise Exception("Expected THEN")
        self.advance()
        true_expr = self.expr()

        if self.current.matches(TT_KEYWORD, 'ELSE'):
            self.advance()
            false_expr = self.expr()
        else:
            false_expr = None

        return IfNode(condition, true_expr, false_expr)

    def while_expr(self):
        self.advance()
        condition = self.expr()
        if not self.current.matches(TT_KEYWORD, 'DO'):
            raise Exception("Expected DO")
        self.advance()
        body = self.expr()
        return WhileNode(condition, body)

    def for_expr(self):
        self.advance()
        var_name = self.current
        self.advance()
        if self.current.type != TT_EQ:
            raise Exception("Expected '=' after variable")
        self.advance()
        start = self.expr()
        if not self.current.matches(TT_KEYWORD, 'TO'):
            raise Exception("Expected TO")
        self.advance()
        end = self.expr()
        if self.current.matches(TT_KEYWORD, 'STEP'):
            self.advance()
            step = self.expr()
        else:
            step = None
        if not self.current.matches(TT_KEYWORD, 'DO'):
            raise Exception("Expected DO")
        self.advance()
        body = self.expr()
        return ForNode(var_name, start, end, step, body)

    def func_def(self):
        self.advance()
        name_tok = None
        if self.current.type == TT_IDENTIFIER:
            name_tok = self.current
            self.advance()

        if self.current.type != TT_LPAREN:
            raise Exception("Expected '('")
        self.advance()

        arg_name_toks = []
        if self.current.type == TT_IDENTIFIER:
            arg_name_toks.append(self.current)
            self.advance()
            while self.current.type == TT_COMMA:
                self.advance()
                if self.current.type != TT_IDENTIFIER:
                    raise Exception("Expected identifier")
                arg_name_toks.append(self.current)
                self.advance()

        if self.current.type != TT_RPAREN:
            raise Exception("Expected ')'")
        self.advance()

        if self.current.type != TT_ARROW:
            raise Exception("Expected '->'")
        self.advance()
        if self.current.matches(TT_KEYWORD, 'RETURN'):
            self.advance()
            expr = self.expr()
            body_node = ReturnNode(expr)
        else:
            body_node = self.expr()
        return FuncDefNode(name_tok, arg_name_toks, body_node)

    def bin_op(self, func, ops):
        left = func()
        while self.current.type in ops or (self.current.type, self.current.value) in ops:
            op_tok = self.current
            self.advance()
            right = func()
            left = BinOpNode(left, op_tok, right)
        return left

class NovaValue:
    def __init__(self):
        self.set_context()

    def set_context(self, ctx=None):
        self.context = ctx
        return self

    def is_true(self):
        return False

    def __repr__(self):
        return str(self)

class NumberVal(NovaValue):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def is_true(self):
        return self.value != 0

    def __repr__(self):
        return str(self.value)

class StringVal(NovaValue):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def is_true(self):
        return len(self.value) > 0

    def __repr__(self):
        return f'"{self.value}"'

class BoolVal(NovaValue):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def is_true(self):
        return self.value

    def __repr__(self):
        return 'TRUE' if self.value else 'FALSE'

class FunctionVal(NovaValue):
    def __init__(self, name, body_node, arg_names):
        super().__init__()
        self.name = name or "<anonymous>"
        self.body_node = body_node
        self.arg_names = arg_names


    def execute(self, args):
        if len(args) != len(self.arg_names):
            raise NovaRuntimeError(None, None, f"{self.name} expected {len(self.arg_names)} args, got {len(args)}", self.context)

        new_context = Context(self.name, self.context)
        new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)

        for i in range(len(args)):
            new_context.symbol_table.set(self.arg_names[i], args[i])

        result = NovaInterpreter().evaluate(self.body_node, new_context)

        if isinstance(self.body_node, ReturnNode):
            return result
        if isinstance(result, ReturnNode):
            return NovaInterpreter().evaluate(result.value_node, new_context)

        return result
    def __repr__(self):
        return f"<func {self.name}>"

class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def get(self, name):
        value = self.symbols.get(name, None)
        if value is None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

class Context:
    def __init__(self, display_name, parent=None):
        self.display_name = display_name
        self.parent = parent
        self.symbol_table = SymbolTable(parent.symbol_table if parent else None)

class NovaInterpreter:
    def visit_ProgramNode(self, node, ctx):
        outputs = []
        for stmt in node.statements:
            result = self.evaluate(stmt, ctx)
            if not isinstance(stmt, VarAssignNode):
                outputs.append(str(result))
        return "\n".join(outputs) if outputs else None
    def evaluate(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f"No method visit_{type(node).__name__}")

    def visit_NumberNode(self, node, ctx):
        return NumberVal(node.tok.value).set_context(ctx)

    def visit_StringNode(self, node, ctx):
        return StringVal(node.tok.value).set_context(ctx)

    def visit_BoolNode(self, node, ctx):
        val = node.tok.value == "TRUE"
        return BoolVal(val).set_context(ctx)

    def visit_VarAccessNode(self, node, ctx):
        val = ctx.symbol_table.get(node.var_name_tok.value)
        if val is None:
            raise NovaRuntimeError(
                node.var_name_tok.start,
                node.var_name_tok.end,
                f"'{node.var_name_tok.value}' is not defined",
                ctx
            )
        return val

    def visit_VarAssignNode(self, node, ctx):
        val = self.evaluate(node.value_node, ctx)
        ctx.symbol_table.set(node.var_name_tok.value, val)
        return val

    def visit_UnaryOpNode(self, node, ctx):
        val = self.evaluate(node.node, ctx)
        if node.op_tok.type == TT_MINUS:
            return NumberVal(-val.value).set_context(ctx)
        elif node.op_tok.matches(TT_KEYWORD, 'NOT'):
            return BoolVal(not val.is_true()).set_context(ctx)
        return val

    def visit_BinOpNode(self, node, ctx):
        left = self.evaluate(node.left, ctx)
        right = self.evaluate(node.right, ctx)
        op = node.op_tok.type

        if isinstance(left, NumberVal) and isinstance(right, NumberVal):
            if op == TT_PLUS: return NumberVal(left.value + right.value).set_context(ctx)
            if op == TT_MINUS: return NumberVal(left.value - right.value).set_context(ctx)
            if op == TT_MUL: return NumberVal(left.value * right.value).set_context(ctx)
            if op == TT_DIV:
                if right.value == 0:
                    raise NovaRuntimeError(node.op_tok.start, node.op_tok.end, "Division by zero", ctx)
                return NumberVal(left.value / right.value).set_context(ctx)
            if op == TT_EE: return BoolVal(left.value == right.value).set_context(ctx)
            if op == TT_NE: return BoolVal(left.value != right.value).set_context(ctx)
            if op == TT_LT: return BoolVal(left.value < right.value).set_context(ctx)
            if op == TT_GT: return BoolVal(left.value > right.value).set_context(ctx)
            if op == TT_LTE: return BoolVal(left.value <= right.value).set_context(ctx)
            if op == TT_GTE: return BoolVal(left.value >= right.value).set_context(ctx)

        if isinstance(left, StringVal) and isinstance(right, StringVal):
            if op == TT_PLUS:
                return StringVal(left.value + right.value).set_context(ctx)

        if op == TT_KEYWORD and node.op_tok.value == 'AND':
            return BoolVal(left.is_true() and right.is_true()).set_context(ctx)
        if op == TT_KEYWORD and node.op_tok.value == 'OR':
            return BoolVal(left.is_true() or right.is_true()).set_context(ctx)

        raise NovaRuntimeError(node.op_tok.start, node.op_tok.end, "Unsupported binary operation", ctx)

    def visit_IfNode(self, node, ctx):
        condition = self.evaluate(node.condition_node, ctx)
        if condition.is_true():
            return self.evaluate(node.true_node, ctx)
        elif node.false_node:
            return self.evaluate(node.false_node, ctx)
        return BoolVal(False).set_context(ctx)

    def visit_WhileNode(self, node, ctx):
        result = None
        counter = 0
        while self.evaluate(node.condition_node, ctx).is_true():
            if counter > 10000:
                raise NovaRuntimeError(
                node.condition_node.op_tok.start if hasattr(node.condition_node, 'op_tok') else None,
                node.condition_node.op_tok.end if hasattr(node.condition_node, 'op_tok') else None,
                "WHILE loop exceeded 10,000 iterations",
                ctx
            )       
            result = self.evaluate(node.body_node, ctx)
            counter += 1
        return result or BoolVal(False).set_context(ctx)

    def visit_ForNode(self, node, ctx):
        start = self.evaluate(node.start_node, ctx).value
        end = self.evaluate(node.end_node, ctx).value
        step = self.evaluate(node.step_node, ctx).value if node.step_node else 1

        i = start
        ctx.symbol_table.set(node.var_name_tok.value, NumberVal(i))
        result = None

        if step >= 0:
            while i <= end:
                ctx.symbol_table.set(node.var_name_tok.value, NumberVal(i))
                result = self.evaluate(node.body_node, ctx)
                i += step
        else:
            while i >= end:
                ctx.symbol_table.set(node.var_name_tok.value, NumberVal(i))
                result = self.evaluate(node.body_node, ctx)
                i += step

        return result or BoolVal(False).set_context(ctx)

    def visit_FuncDefNode(self, node, ctx):
        func_name = node.name_tok.value if node.name_tok else None
        arg_names = [arg.value for arg in node.arg_name_toks]
        func_val = FunctionVal(func_name, node.body_node, arg_names).set_context(ctx)

        if node.name_tok:
            ctx.symbol_table.set(func_name, func_val)

        return func_val
    
    def visit_ReturnNode(self, node, ctx):
        return self.evaluate(node.value_node, ctx)

    def visit_CallNode(self, node, ctx):
        func = self.evaluate(node.node_to_call, ctx)
        args = [self.evaluate(arg, ctx) for arg in node.arg_nodes]
        return func.execute(args)
    def visit_CallNode(self, node, ctx):
        func = self.evaluate(node.node_to_call, ctx)
        args = [self.evaluate(arg, ctx) for arg in node.arg_nodes]

        if not isinstance(func, FunctionVal):
            raise NovaRuntimeError(None, None, "Tried to call a non-function", ctx)

        return func.execute(args)

def run_nova(file_name, code, context=None, interpreter=None):
    lexer = NovaLexer(file_name, code)
    tokens, err = lexer.tokenize()
    if err:
        return None, err.describe()

    parser = NovaParser(tokens)
    try:
        tree = parser.parse()
    except Exception as e:
        return None, f"SyntaxError: {str(e)}"

    if context is None:
        context = Context('<main>')
    if interpreter is None:
        interpreter = NovaInterpreter()

    try:
        result = interpreter.evaluate(tree, context)
        return result, None
    except NovaRuntimeError as e:
        return None, e.describe()

