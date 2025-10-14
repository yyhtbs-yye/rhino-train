import psutil
import os
import gc
import torch

def count_objects_by_type():
    """Count objects by type to identify which objects are increasing."""
    
    # Get all objects
    all_objects = gc.get_objects()
    
    # Group by type
    type_counts = {}
    for obj in all_objects:
        obj_type = type(obj).__name__
        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
    
    # Sort by count (highest first)
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Print top types
    print("Top object types:")
    for obj_type, count in sorted_types[:20]:  # Show top 20
        print(f"  {obj_type}: {count}")
    
    return sorted_types

def track_object_count_changes(previous_counts=None):
    """Track changes in object counts between calls."""
    import gc
    
    # Get current counts
    current_counts = {}
    for obj in gc.get_objects():
        obj_type = type(obj).__name__
        current_counts[obj_type] = current_counts.get(obj_type, 0) + 1
    
    # Compare with previous counts if available
    if previous_counts:
        print("Object count changes:")
        
        # Find types with significant increases
        significant_changes = []
        for obj_type, count in current_counts.items():
            prev_count = previous_counts.get(obj_type, 0)
            change = count - prev_count
            if change > 10:  # Arbitrary threshold for significant change
                significant_changes.append((obj_type, count, change))
        
        # Sort by absolute change
        significant_changes.sort(key=lambda x: x[2], reverse=True)
        
        # Print significant changes
        for obj_type, count, change in significant_changes[:15]:  # Top 15 changes
            print(f"  {obj_type}: {prev_count} -> {count} (+{change})")
    
    return current_counts

def find_referrers_to_type(target_type_name, limit=5):
    """Find objects that refer to instances of the specified type."""
    import gc
    import inspect
    
    # Find all instances of the target type
    target_instances = [obj for obj in gc.get_objects() 
                      if type(obj).__name__ == target_type_name]
    
    if not target_instances:
        print(f"No instances of {target_type_name} found.")
        return
    
    print(f"Found {len(target_instances)} instances of {target_type_name}")
    
    # Take a sample of instances to analyze
    sample_instances = target_instances[:min(limit, len(target_instances))]
    
    for i, instance in enumerate(sample_instances):
        print(f"\nReferrers to {target_type_name} #{i+1}:")
        
        # Find all referrers to this instance
        referrers = gc.get_referrers(instance)
        
        for j, referrer in enumerate(referrers[:10]):  # Limit to 10 referrers
            referrer_type = type(referrer).__name__
            
            # Handle different referrer types
            if referrer_type == 'dict':
                # For dictionaries, find if it's part of an object
                dict_keys = list(referrer.keys())[:5]  # Show first 5 keys
                dict_owners = [obj for obj in gc.get_objects() 
                             if hasattr(obj, '__dict__') and obj.__dict__ is referrer]
                
                if dict_owners:
                    owner_types = [type(owner).__name__ for owner in dict_owners]
                    print(f"  Referrer #{j+1}: __dict__ of {owner_types}")
                else:
                    print(f"  Referrer #{j+1}: dict with keys {dict_keys}...")
            
            elif referrer_type == 'list':
                print(f"  Referrer #{j+1}: list of length {len(referrer)}")
                # Try to find owner of the list
                list_owners = [obj for obj in gc.get_objects() 
                             if isinstance(obj, object) and 
                             any(getattr(obj, attr, None) is referrer 
                                for attr in dir(obj) 
                                if not attr.startswith('__'))]
                
                if list_owners:
                    owner_types = [type(owner).__name__ for owner in list_owners]
                    print(f"    Possibly owned by: {owner_types}")
            
            elif inspect.isframe(referrer):
                frame_info = inspect.getframeinfo(referrer)
                print(f"  Referrer #{j+1}: frame at {frame_info.filename}:{frame_info.lineno}")
            
            else:
                print(f"  Referrer #{j+1}: {referrer_type}")

def track_object_increases(func):
    """Decorator to track object count increases during a function call."""
    import gc
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Count objects before
        before_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            before_counts[obj_type] = before_counts.get(obj_type, 0) + 1
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Count objects after
        after_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            after_counts[obj_type] = after_counts.get(obj_type, 0) + 1
        
        # Find significant increases
        print(f"Object count changes during {func.__name__}:")
        significant_changes = []
        for obj_type in after_counts:
            before = before_counts.get(obj_type, 0)
            after = after_counts[obj_type]
            change = after - before
            if change > 5:  # Threshold for reporting
                significant_changes.append((obj_type, before, after, change))
        
        significant_changes.sort(key=lambda x: x[3], reverse=True)
        
        for obj_type, before, after, change in significant_changes[:10]:
            print(f"  {obj_type}: {before} -> {after} (+{change})")
        
        return result
    
    return wrapper

def log_detailed_memory():
    process = psutil.Process(os.getpid())
    ram = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
    
    # Add GPU memory tracking if relevant
    gpu_mem = f", GPU: {torch.cuda.memory_allocated() / (1024 * 1024 * 1024):.2f} GB" if torch.cuda.is_available() else ""
    
    # Count objects to detect Python object leaks
    total_objects = len(gc.get_objects())
    
    # Count tensors - this is very useful for PyTorch memory leaks
    tensor_count = sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor))
    
    print(f"RAM: {ram:.2f} GB{gpu_mem}, Objects: {total_objects}, Tensors: {tensor_count}")
    
    return ram, tensor_count
class MemoryDebugCallback:
    def __init__(self):
        self.previous_counts = None
        self.batch_idx = 0
        
    def on_batch_end(self, trainer, boat, batch, batch_idx, loss):
        """Track memory after each batch."""
        import gc
        
        self.batch_idx += 1
        
        # Only check every few batches to reduce overhead
        if self.batch_idx % 5 == 0:
            print(f"\n==== Memory Debug after batch {self.batch_idx} ====")
            
            # Count total objects
            total_objects = len(gc.get_objects())
            print(f"Total Python objects: {total_objects}")
            
            # Track changes by type
            self.previous_counts = track_object_count_changes(self.previous_counts)
            
            # Look for potential memory leaks
            gc.collect()  # Force garbage collection
            
            # Count again after garbage collection
            total_after_gc = len(gc.get_objects())
            print(f"After GC: {total_after_gc} objects ({total_objects - total_after_gc} freed)")

