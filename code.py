# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:47:46 2024

@author: keertanapriya
"""

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def is_operator(char):
    return char in ['+', '-', '*', '/']

def construct_parse_tree(postfix_expression):
    stack = []
    for token in postfix_expression.split():
        if not is_operator(token):
            stack.append(TreeNode(token))
        else:
            if len(stack) < 2:
                raise ValueError("Invalid postfix expression")
            right_node = stack.pop()
            left_node = stack.pop()
            operator_node = TreeNode(token)
            operator_node.left = left_node
            operator_node.right = right_node
            stack.append(operator_node)
    if len(stack) != 1:
        raise ValueError("Invalid postfix expression")
    return stack[0]

def print_parse_tree(node, level=0, position="Root"):
    if node is not None:
        print('  ' * level + position + ": " + str(node.value))
        print_parse_tree(node.left, level + 1, "Left")
        print_parse_tree(node.right, level + 1, "Right")

def infix_to_postfix(expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}

    output = []
    operator_stack = []

    for token in expression.split():
        if token.isdigit():
            output.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output.append(operator_stack.pop())
            operator_stack.pop()  
        else: 
            while operator_stack and precedence.get(token, 0) <= precedence.get(operator_stack[-1], 0):
                output.append(operator_stack.pop())
            operator_stack.append(token)
    while operator_stack:
        output.append(operator_stack.pop())
    return ' '.join(output)

infix_expression = "3 + 4 * ( 2 - 1  + 7 )"
postfix_expression = infix_to_postfix(infix_expression)
parse_tree = construct_parse_tree(postfix_expression)
print("Parse tree:")
print_parse_tree(parse_tree)


# def infix_to_postfix(expression):
#     precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
#     associativity = {'NOT': 'right', 'AND': 'left', 'OR': 'left'}
#     output = []
#     operator_stack = []
#     tokens = expression.split()
#     for token in tokens:
#         if token not in ['NOT', 'AND', 'OR']: 
#             output.append(token)
#         else:
#             while (operator_stack and
#                    precedence.get(token, 0) <= precedence.get(operator_stack[-1], 0) and
#                    (associativity[token] == 'left' or precedence[token] < precedence[operator_stack[-1]])):
#                 output.append(operator_stack.pop())
#             operator_stack.append(token)
#     while operator_stack:
#         output.append(operator_stack.pop())
#     return ' '.join(output)

# def generate_temp(temp_count):
#     temp = f"t{temp_count[0]}"
#     temp_count[0] += 1
#     return temp

# def generate_code(expression):
#     code = {}
#     temp_count = [0]
#     stack = []
#     tokens = expression.split()
#     for token in tokens:
#         if token == 'NOT':  
#             op = stack.pop()
#             temp = generate_temp(temp_count)
#             code[temp]=('NOT', op)
#             stack.append(temp)
#         elif token in ['AND', 'OR']:  
#             op2 = stack.pop()
#             op1 = stack.pop()
#             temp = generate_temp(temp_count)
#             code[temp]=(token, op1, op2)
#             stack.append(temp)
#         else:  
#             stack.append(token)
#     for k,v in code.items():
#         print(k,':',v)

# expression = "NOT b OR c AND d"
# postfix_expression = infix_to_postfix(expression)
# print(postfix_expression)
# generate_code(postfix_expression)


------------------------------------------

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __str__(self):
        return str(self.value)
    
class ThreeAddressCodeGenerator:
    def __init__(self):
        self.code = []
        self.temp_count = 0
        self.root = None

    def generate_temp(self):
        temp = f"t{self.temp_count}"
        self.temp_count += 1
        return temp

    def generate_code(self, expression):
        lines=[]
        stack = []
        tokens = expression.split()
        for token in tokens:
            if token.isalnum():  
                stack.append(token)
            else: 
                op2 = stack.pop()
                op1 = stack.pop()
                temp = self.generate_temp()
                self.code.append((token, op1, op2, temp))
                stack.append(temp)
        for line in self.code:
            lines.append(line)
        return lines
            
    def generate_parse_tree(self, code):
        self.root = self._construct_tree(code)

    def _construct_tree(self, code):
        stack = []
        for op, op1, op2, temp in code:
            node = TreeNode((op, temp))
            if op1.isdigit() or op1.isalpha():
                node.add_child(TreeNode(op1))
            else:
                node.add_child(stack.pop())

            if op2.isdigit() or op2.isalpha():
                node.add_child(TreeNode(op2))
            else:
                node.add_child(stack.pop())
            stack.append(node)
        return stack.pop()

    def print_tree(self, node, level=0):
        if node:
            print("  " * level + str(node))
            for child in node.children:
                self.print_tree(child, level + 1)
            
def infix_to_postfix(expression):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    output = []
    operator_stack = []
    for token in expression.split():
        if token.isdigit():
            output.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output.append(operator_stack.pop())
            operator_stack.pop()  
        else: 
            while operator_stack and precedence.get(token, 0) <= precedence.get(operator_stack[-1], 0):
                output.append(operator_stack.pop())
            operator_stack.append(token)
    while operator_stack:
        output.append(operator_stack.pop())
    return ' '.join(output)

infix_expression = "3 + 4 * ( 2 - 1 )"
postfix_expression = infix_to_postfix(infix_expression)
generator = ThreeAddressCodeGenerator()
lines=generator.generate_code(postfix_expression)
generator.generate_parse_tree(lines)
generator.print_tree(generator.root)
--------------------------------------

# slr parsing

from collections import defaultdict
from prettytable import PrettyTable

def slr(s, parsing_table, n_nt, stack, action_l):
    table = PrettyTable()
    table.field_names = ["Stack", "Action", "Input String"]

    while len(s):
        action = parsing_table[stack[-1]][s[0]]
        
        if action == "ACCEPT":
            action_l.append(action)
            print("ACCEPTED")
            break
        if action == "-":
            action_l.append("not accepted")
            print("REJECTED")
            break

        s_shift = action.split(":")
        action_l.append(s_shift[0] + " " + s_shift[1])

        if s_shift[0] == "Shift":
            stack.append(s[0])
            stack.append(s_shift[1])
            s = s[1:]
        elif s_shift[0] == "Reduce":
            s_reduce = s_shift[1].split("->")
            replace_string = ""
            stack_replace_string = ""

            for i in range(len(stack) - 1, 0, -1):
                stack_replace_string += stack[i]
                if stack[i] in n_nt:
                    replace_string += stack[i]
                rev_s = replace_string[::-1]
                if rev_s == s_reduce[1]:
                    break

            for i in range(len(stack_replace_string)):
                stack.pop()
            stack.append(s_reduce[0])
            stack.append(parsing_table[stack[-2]][stack[-1]])

        table.add_row(["".join(stack), action_l[-1], s])
    print("\n\nPARSING TABLE\n")
    print(table)


with open("slr_table.txt",'r') as file:
    f=file.readlines()

parsing_table = defaultdict(dict)
n_nt = []

for line in f:
    x = line.strip('\n')
    y = x.split()
    if y[0] == "state":
        n_nt = y[1:]
        continue
    for i in range(len(n_nt)):
        parsing_table[y[0]][n_nt[i]] = y[i + 1]



s = input("Enter input string: ")
s += "$"
stack = ['0']
action_l = []
slr(s, parsing_table, n_nt,stack,action_l)

--------------------------------------------------
# from prettytable import PrettyTable

# exp = input("Enter arithmetic expression with appropriate brackets: ")
# print("\nARITHMETIC EXPRESSION: ", exp)

# operators = ['*', '+', '-', '/']
# asg = exp[0]
# stack = list(exp[2:])

# int_code = {}
# count = 0

# for i in range(len(exp)):
#     if exp[i] == ')':
#         s = ''
#         while stack[-1] != '(':
#             s = stack.pop() + s
#         stack.pop()  # Remove the '('

#         count += 1
#         temp_var = 'T' + str(count)
#         int_code[temp_var] = s
#         stack.append(temp_var)
#     else:
#         stack.append(exp[i])

# int_code[asg] = 'T' + str(count)

# print("\n\nFinal Three Address Code:\n")
# for key, value in int_code.items():
#     print(key, '=', value)

# op = []
# arg1 = []
# arg2 = []
# res = []

# print("\n\nQUADRUPLES\n")
# for k, v in int_code.items():
#     if v[0] == '-':
#         op.append(v[0])
#         arg1.append(v[1:])
#         arg2.append('-')
#         res.append(k)
#     else:
#         flag = 0
#         for i in operators:
#             if i in v:
#                 flag = 1
#                 ind = v.index(i)
#                 op.append(v[ind])
#                 arg1.append(v[0:ind])
#                 arg2.append(v[ind + 1:])
#                 res.append(k)
#                 break
#         if flag == 0:
#             op.append('=')
#             arg1.append(v)
#             arg2.append('-')
#             res.append(k)

# quad_table = PrettyTable(['OPERATOR', 'ARG1', 'ARG2', 'RESULT'])

# for i in range(len(op)):
#     quad_table.add_row([''.join(op[i]), ''.join(arg1[i]), ''.join(arg2[i]), ''.join(res[i])])

# print(quad_table)

# print("\n\nTRIPLES\n")

# op_triples = []
# arg1_triples = []
# arg2_triples = []
# count_triples = 1

# for i in range(len(op)):
#     if op[i] != '=':  # Exclude assignments
#         op_triples.append(op[i])
#         if arg1[i][0] == 'T':
#             arg1_triples.append(str(list(int_code.keys()).index(arg1[i]) + 1))
#         else:
#             arg1_triples.append(arg1[i])
#         if arg2[i][0] == 'T':
#             arg2_triples.append(str(list(int_code.keys()).index(arg2[i]) + 1))
#         else:
#             arg2_triples.append(arg2[i])

# triple_table = PrettyTable(['S_No', 'Operator', 'Argument 1', 'Argument 2'])

# for i in range(len(op_triples)):
#     triple_table.add_row([count_triples, op_triples[i], arg1_triples[i], arg2_triples[i]])
#     count_triples += 1

# print(triple_table)
------------------------------------------

# def takeGrammar():
#   grammar = {}

#   production = ''
#   while production != '#':
#     production = input("Enter the production : ")
#     currProd = production.split("->")
#     lhs = currProd[0]
#     rhs = ''.join(currProd[1:]).split('/')
#     grammar[lhs] = rhs
#   return grammar


grammar = {'S': ['AA'], 'A': ['aA', 'b'], '#': ['']}
print(grammar)

# find the augmented grammar
augGrammar = {'S\'' : ['.S']}
for lhs, rhs in grammar.items():
  augmented = []
  if lhs != '#':
    for item in rhs:
      augmented.append('.' + item)
    augGrammar[lhs] = augmented

print(augGrammar)

def Generatelr0(grammar):
    items = []
    for lhs, rhsList in grammar.items():
        for rhs in rhsList:
            for position in range(len(rhs) + 1):
                item = (lhs, rhs[:position] + '.' + rhs[position:])
                items.append(item)
    return items

grammar = {'S\'' : ['S'], 'S': ['AA'], 'A': ['aA', 'b']}
lr0Items = Generatelr0(grammar)
print(lr0Items)

def closure(items, grammar):
    closureSet = set(items)
    added = True
    while added:
        added = False
        for (lhs, rhs) in list(closureSet):
            dot_position = rhs.find('.')
            if dot_position < len(rhs) - 1:
                next_symbol = rhs[dot_position + 1]
                for production in grammar.get(next_symbol, []):
                    new_item = (next_symbol, '.' + production)
                    if new_item not in closureSet:
                        closureSet.add(new_item)
                        added = True
    return closureSet

def goto(items, symbol, grammar):
    gotoSet = set()
    for (lhs, rhs) in items:
        dotPosition = rhs.find('.')
        if dotPosition < len(rhs) - 1 and rhs[dotPosition + 1] == symbol:
            movedItem = (lhs, rhs[:dotPosition] + symbol + '.' + rhs[dotPosition + 2:])
            gotoSet.add(movedItem)
    return closure(gotoSet, grammar)

def construct_dfa(start_symbol, grammar):
    initial_item = (start_symbol, '.' + grammar[start_symbol][0])
    initial_state = closure({initial_item}, grammar)

    states = [initial_state]
    transitions = [] # Each transition is a tuple (from_state, symbol, to_state)
    state_index = {tuple(initial_state): 0} # Map state sets to state indexes

    queue = [initial_state] # Queue for BFS
    while queue:
        current_state = queue.pop(0)
        current_index = state_index[tuple(current_state)]

        # For each symbol in the grammar, try to apply goto
        symbols = set(symbol for item in current_state for symbol in item[1] if symbol != '.')
        for symbol in symbols:
            next_state = goto(current_state, symbol, grammar)
            if not next_state:
                continue # Skip if no transition occurs

            if tuple(next_state) not in state_index:
                # Found a new state
                state_index[tuple(next_state)] = len(states)
                states.append(next_state)
                queue.append(next_state)

            next_index = state_index[tuple(next_state)]
            transitions.append((current_index, symbol, next_index))

    return states, transitions

start_symbol = "S'"

dfa_states, dfa_transitions = construct_dfa(start_symbol, grammar)

# To display the DFA states and transitions
print(f"States (number of states: {len(dfa_states)}):")
for i, state in enumerate(dfa_states):
    print(f"State {i}: {state}")

print("\nTransitions:")
for transition in dfa_transitions:
    print(f"From state {transition[0]} to state {transition[2]} via symbol '{transition[1]}'")

def compute_first_sets(grammar):
    first_sets = {key: set() for key in grammar}
    changed = True

    while changed:
        changed = False
        for nonterminal, productions in grammar.items():
            for production in productions:
                # For ε-productions
                if production == 'ε':
                    if 'ε' not in first_sets[nonterminal]:
                        first_sets[nonterminal].add('ε')
                        changed = True
                else:
                    for symbol in production:
                        if symbol in grammar:  # Non-terminal
                            original_len = len(first_sets[nonterminal])
                            first_sets[nonterminal].update(first_sets[symbol] - {'ε'})
                            if 'ε' not in first_sets[symbol]:
                                break
                            if original_len != len(first_sets[nonterminal]):
                                changed = True
                        else:  # Terminal
                            if symbol not in first_sets[nonterminal]:
                                first_sets[nonterminal].add(symbol)
                                changed = True
                            break
    return first_sets

def compute_follow_sets(grammar, first_sets, start_symbol):
    follow_sets = {key: set() for key in grammar}
    follow_sets[start_symbol].add('$')  # End of input symbol for start symbol

    changed = True
    while changed:
        changed = False
        for nonterminal, productions in grammar.items():
            for production in productions:
                follow_temp = follow_sets[nonterminal]
                for symbol in reversed(production):
                    if symbol in follow_sets:
                        original_len = len(follow_sets[symbol])
                        follow_sets[symbol].update(follow_temp)
                        if 'ε' in first_sets[symbol]:
                            follow_temp = follow_temp.union(first_sets[symbol] - {'ε'})
                        else:
                            follow_temp = first_sets[symbol]
                        if original_len != len(follow_sets[symbol]):
                            changed = True
                    else:
                        follow_temp = {symbol}
    return follow_sets

def construct_slr_table(grammar, dfa_states, dfa_transitions, start_symbol):
    action_table = {}
    goto_table = {}
    first_sets = compute_first_sets(grammar)
    follow_sets = compute_follow_sets(grammar, first_sets, start_symbol)

    for i, state in enumerate(dfa_states):
        for item in state:
            lhs, rhs = item
            if '.' not in rhs[-1]: # A production can be reduced
                for symbol in follow_sets[lhs]:
                    action_table[(i, symbol)] = ('reduce', lhs, rhs.replace('.', ''))
            if rhs[-1] == '.': # Accept condition for augmented grammar
                if lhs == start_symbol:
                    action_table[(i, '$')] = ('accept',)

        for transition in dfa_transitions:
            from_state, symbol, to_state = transition
            if from_state == i:
                if symbol.isupper(): # A goto for a non-terminal
                    goto_table[(i, symbol)] = to_state
                else: # A shift for a terminal
                    action_table[(i, symbol)] = ('shift', to_state)


    return action_table, goto_table

action_table, goto_table = construct_slr_table(grammar, dfa_states, dfa_transitions, start_symbol)

# Displaying the tables
print("Action Table:")
for key, value in sorted(action_table.items()):
    print(f"State {key[0]}, Symbol '{key[1]}': {value}")

print("\nGoto Table:")
for key, value in sorted(goto_table.items()):
    print(f"State {key[0]}, Symbol '{key[1]}': Goto State {value}")


print(action_table)
