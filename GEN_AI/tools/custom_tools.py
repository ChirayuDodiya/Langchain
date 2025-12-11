from langchain_core.tools import tool

def multiply(a,b):
    """multiply two numbers"""
    return a*b

def multiply(a:int,b:int) ->int:
    """multiply two numbers"""
    return a*b

@tool
def multiply(a:int,b:int) ->int:
    """multiply two numbers"""
    return a*b

result = multiply.invoke({"a":3,"b":4})
print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.args_schema.model_json_schema())