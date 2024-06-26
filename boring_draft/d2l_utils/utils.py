import inspect
import pprint

# color patterns
RED_PATTERN = '\033[31m%s\033[0m'
GREEN_PATTERN = '\033[32m%s\033[0m'
BLUE_PATTERN = '\033[34m%s\033[0m'
PEP_PATTERN = '\033[36m%s\033[0m'
BROWN_PATTERN = '\033[33m%s\033[0m'
YELLOW_PATTERN = '\033[93m%s\033[0m'


def print_debug_info(*args, **kwargs):
    # Print information for positional arguments
    for i, arg in enumerate(args):
        print(f"Positional arg {i} {arg}: type={type(arg)}", end="")
        if isinstance(arg, torch.Tensor):
            print(f", dtype={arg.dtype}, device={arg.device}")
        else:
            print()
    
    # Print information for keyword arguments
    for key, value in kwargs.items():
        print(f"Keyword arg {key} {value}: type={type(value)}", end="")
        if isinstance(value, torch.Tensor):
            print(f", dtype={value.dtype}, device={value.device}")
        elif hasattr(value, 'device'):  # For custom objects that might have a 'device' attribute
            print(f", device={value.device}")
        else:
            print()


def mprint(obj, magic_methods=False, private_methods=True, public_methods=True):
    # Split items based on their types
    
    if private_methods:
        magic_methods = [x for x in dir(obj) if x.startswith("__") and x.endswith("__")]
        private_methods = [x for x in dir(obj) if x.startswith("_") and not x in magic_methods]
        print("\n\033[93mPrivate Methods:\033[0m")
        for item in sorted(private_methods):
            print(f"    {item}")

        if magic_methods:
            print("\n\033[93mMagic Methods:\033[0m")
            for item in sorted(magic_methods):
                print(f"    {item}")
    
    if public_methods:
        public_methods = [x for x in dir(obj) if not x.startswith("_")]
        print("\n\033[93mPublic Methods:\033[0m")
        for item in sorted(public_methods):
            print(f"    {item}")


def cprint(*exprs, c=None):
    """
    Custom print function that prints the name of the variable/expression
    alongside its value.
    
    Parameters:
    - *exprs: Variable-length argument list of expressions to evaluate.
    - globals, locals (dict, optional): Optional dictionaries to specify the namespace 
      for the evaluation. This allows the function to access variables outside of its 
      local scope.
    
    Example:
    x = 10
    y = 20
    cprint(x)
    # Output: x: 10
    
    cprint(x, y)
    # Output:
    # x: 10
    # y: 20
    
    cprint()
    # Output: (Empty line)
    """
    # Fetch the line of code that called cprint
    frame = inspect.currentframe().f_back
    call_line = inspect.getframeinfo(frame).code_context[0].strip()
    
    # Extract the arguments from the line
    arg_str = call_line[call_line.index('(') + 1:-1].strip()
    
    # Split the arguments by comma, keeping expressions intact
    arg_list = []
    bracket_count = 0
    current_arg = []
    for char in arg_str:
        if char == ',' and bracket_count == 0:
            arg_list.append(''.join(current_arg).strip())
            current_arg = []
        else:
            if char in '([{':
                bracket_count += 1
            elif char in ')]}':
                bracket_count -= 1
            current_arg.append(char)
    if current_arg:
        arg_list.append(''.join(current_arg).strip())
    
    # Check if there are any arguments
    if not arg_list or (len(arg_list) == 1 and not arg_list[0]):
        print()  # Print an empty line
        return
    
    for arg, expr in zip(arg_list, exprs):
        try:
            if not c:
                print(YELLOW_PATTERN % f"{arg}:")
                pprint.pprint(expr)
            if c == 'red':
                print(RED_PATTERN % f"{arg}:")
                pprint.pprint(expr)
            elif c == 'green':
                print(GREEN_PATTERN % f"{arg}:")
                pprint.pprint(expr)
            elif c == 'blue':
                print(BLUE_PATTERN % f"{arg}:")
                pprint.pprint(expr)
            elif c == 'pep':
                print(PEP_PATTERN % f"{arg}:")
                pprint.pprint(expr)
            elif c == 'brown':
                print(BROWN_PATTERN % f"{arg}:")
                pprint.pprint(expr)
            elif c == 'normal':
                pprint.pprint(arg)

        except Exception as e:
            print(f"Error evaluating {arg}: {e}")


def sprint(*exprs, globals=None, locals=None):
    """
    Custom print function that prints the name of the variable/expression
    alongside its value.
    
    Parameters:
    - *exprs (str): Variable-length argument list of expressions to evaluate as strings.
    - globals, locals (dict, optional): Optional dictionaries to specify the namespace
      for the evaluation. This allows the function to access variables outside of its
      local scope.
    
    Example:
    x = 10
    y = 20
    sprint("x")
    # Output: x: 10
    
    sprint("x", "y")
    # Output:
    # x: 10
    # y: 20
    
    sprint()
    # Output: (Empty line)
    """
    # Check if there are any arguments
    if not exprs:
        print()  # Print an empty line
        return
    
    for expr in exprs:
        try:
            # Evaluate the expression
            value = eval(expr, globals, locals)
            print(f"\033[93m{expr}\033[0m: \n{value}\n")
        except Exception as e:
            print(f"Error evaluating {expr}: {e}")


# import torch 

# def count_unique(tensor):
#     # Calculate unique values and their counts
#     unique_values, counts = torch.unique(tensor, return_counts=True)

#     # Convert unique_values to a Python list
#     unique_values = unique_values.tolist()

#     # Convert counts to a Python list
#     counts = counts.tolist()

#     # Print the unique values and their counts
#     for value, count in zip(unique_values, counts):
#         print(f"Value: {value}, Count: {count}")
# 
#     print()
