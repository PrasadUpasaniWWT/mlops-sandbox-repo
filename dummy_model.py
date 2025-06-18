# dummy_model.py

def load():
    # This is a dummy load; no model artifacts needed.
    return None

def invoke(model, request):
    """
    Expects input like: "1, 2, 3, 4, 5"
    Returns: 15
    """
    try:
        numbers = list(map(float, request.strip().split(',')))
        return {"sum": sum(numbers)}
    except Exception as e:
        return {"error": str(e)}
