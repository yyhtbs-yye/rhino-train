import psutil

class GlobalStep:
    def __init__(self, initial_value):
        self.value = initial_value
    
    def __call__(self):
        return self.value
    
    def __iadd__(self, other):
        if isinstance(other, int):
            self.value += other
        else:
            raise ValueError("GlobalStep can only be incremented by an integer.")
        return self
    
    def __isub__(self, other):
        if isinstance(other, int):
            self.value -= other
        else:
            raise ValueError("GlobalStep can only be decremented by an integer.")
        return self

def get_ram_info():
    # Get RAM statistics
    ram = psutil.virtual_memory()
    # Return used memory in GB and percentage
    return f"RAM: {ram.used/1073741824:.1f}GB/{ram.total/1073741824:.1f}GB ({ram.percent}%)"
