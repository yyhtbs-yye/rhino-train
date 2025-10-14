class MemorySnapshot:
    """Utility to take memory snapshots and compare them over time."""
    
    def __init__(self):
        import gc
        import psutil
        import sys
        import torch
        
        self.snapshots = []
        self.gc = gc
        self.psutil = psutil
        self.sys = sys
        self.torch = torch
        
    def take_snapshot(self, label=""):
        """Take a snapshot of current memory state."""
        self.gc.collect()  # Force collection to get accurate counts
        
        # Get process memory info
        process = self.psutil.Process()
        ram_usage = process.memory_info().rss
        
        # Count objects by type
        objects = self.gc.get_objects()
        type_counts = {}
        for obj in objects:
            obj_type = type(obj).__name__
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        
        # Count tensors (if using PyTorch)
        tensor_count = 0
        tensor_memory = 0
        if hasattr(self, 'torch'):
            tensors = [obj for obj in objects if isinstance(obj, self.torch.Tensor)]
            tensor_count = len(tensors)
            tensor_memory = sum(t.nelement() * t.element_size() for t in tensors if hasattr(t, 'nelement'))
        
        # Store snapshot
        snapshot = {
            'label': label,
            'time': self.psutil.time.time(),
            'ram_usage': ram_usage,
            'total_objects': len(objects),
            'type_counts': type_counts,
            'tensor_count': tensor_count,
            'tensor_memory': tensor_memory
        }
        
        self.snapshots.append(snapshot)
        return len(self.snapshots) - 1  # Return snapshot index
    
    def compare_snapshots(self, index1, index2=None):
        """Compare two snapshots and print differences."""
        if index2 is None:
            index2 = len(self.snapshots) - 1  # Compare with latest
        
        s1 = self.snapshots[index1]
        s2 = self.snapshots[index2]
        
        print(f"Comparing {s1['label']} (#{index1}) to {s2['label']} (#{index2})")
        
        # Time difference
        time_diff = s2['time'] - s1['time']
        print(f"Time elapsed: {time_diff:.2f} seconds")
        
        # Memory usage difference
        ram_diff = s2['ram_usage'] - s1['ram_usage']
        ram_diff_mb = ram_diff / (1024 * 1024)
        ram_pct = (ram_diff / s1['ram_usage']) * 100 if s1['ram_usage'] > 0 else 0
        print(f"RAM: {s1['ram_usage']/(1024*1024):.2f}MB -> {s2['ram_usage']/(1024*1024):.2f}MB ({ram_diff_mb:+.2f}MB, {ram_pct:+.2f}%)")
        
        # Object count difference
        obj_diff = s2['total_objects'] - s1['total_objects']
        obj_pct = (obj_diff / s1['total_objects']) * 100 if s1['total_objects'] > 0 else 0
        print(f"Total objects: {s1['total_objects']} -> {s2['total_objects']} ({obj_diff:+d}, {obj_pct:+.2f}%)")
        
        # Tensor difference if available
        if 'tensor_count' in s1 and 'tensor_count' in s2:
            tensor_diff = s2['tensor_count'] - s1['tensor_count']
            tensor_pct = (tensor_diff / s1['tensor_count']) * 100 if s1['tensor_count'] > 0 else 0
            print(f"Tensors: {s1['tensor_count']} -> {s2['tensor_count']} ({tensor_diff:+d}, {tensor_pct:+.2f}%)")
            
            tensor_mem_diff = s2['tensor_memory'] - s1['tensor_memory']
            tensor_mem_mb = tensor_mem_diff / (1024 * 1024)
            print(f"Tensor memory: {s1['tensor_memory']/(1024*1024):.2f}MB -> {s2['tensor_memory']/(1024*1024):.2f}MB ({tensor_mem_mb:+.2f}MB)")
        
        # Type count differences (show most significant increases)
        print("\nTop increasing types:")
        type_diffs = []
        
        for type_name in set(s2['type_counts'].keys()) | set(s1['type_counts'].keys()):
            count1 = s1['type_counts'].get(type_name, 0)
            count2 = s2['type_counts'].get(type_name, 0)
            diff = count2 - count1
            if diff > 0:  # Only show increases
                type_diffs.append((type_name, count1, count2, diff))
        
        # Sort by absolute increase
        type_diffs.sort(key=lambda x: x[3], reverse=True)
        
        for type_name, count1, count2, diff in type_diffs[:20]:  # Show top 20
            pct = (diff / count1) * 100 if count1 > 0 else float('inf')
            print(f"  {type_name}: {count1} -> {count2} (+{diff}, +{pct:.2f}%)")
        
        return type_diffs
    
def analyze_leaking_type(type_name):
    """Analyze a leaking object type to find where it's being referenced."""
    import gc
    
    print(f"\nAnalyzing leaking type: {type_name}")
    
    # Find instances of this type
    instances = [obj for obj in gc.get_objects() if type(obj).__name__ == type_name]
    # print(f"Found {len(instances)} instances")
    
    if not instances:
        return
    
    if type_name == "Tensor":
        print("Tensors in CPU -> ", [it.device.type for it in instances].count('cpu'))
        print("Tensors in CUDA -> ", [it.device.type for it in instances].count('cuda'))
    

    print(f"Found {len(instances)} instances")
    return
    # Take a sample
    sample_size = min(5, len(instances))
    sample = instances[:sample_size]
    
    for i, obj in enumerate(sample):
        print(f"\nSample {i+1}:")
        
        # Try to get meaningful info about the object
        try:
            if hasattr(obj, "__dict__"):
                attrs = list(obj.__dict__.keys())[:10]  # First 10 attributes
                print(f"  Attributes: {attrs}")
            
            if hasattr(obj, "__len__"):
                try:
                    length = len(obj)
                    print(f"  Length: {length}")
                except:
                    pass
        except:
            pass
        
        # Find what's referencing this object
        referrers = gc.get_referrers(obj)
        print(f"  Referenced by {len(referrers)} objects")
        
        # Show a few referrer details
        for j, ref in enumerate(referrers[:3]):  # First 3 referrers
            ref_type = type(ref).__name__
            
            print(f"    Referrer {j+1} type: {ref_type}")
            
            # Handle common referrer types
            if ref_type == 'dict':
                # Try to find if it's an object's __dict__
                for potential_owner in gc.get_objects():
                    if hasattr(potential_owner, "__dict__") and potential_owner.__dict__ is ref:
                        print(f"      Part of: {type(potential_owner).__name__} __dict__")
                        break
            
            elif ref_type == 'list':
                # Try to find the list's owner
                for potential_owner in gc.get_objects():
                    if hasattr(potential_owner, "__dict__"):
                        for attr, value in potential_owner.__dict__.items():
                            if value is ref:
                                print(f"      Part of: {type(potential_owner).__name__}.{attr}")
                                break
            
            elif ref_type == 'function':
                if hasattr(ref, "__qualname__"):
                    print(f"      Function: {ref.__module__}.{ref.__qualname__}")
            
            elif ref_type == 'frame':
                import inspect
                frame_info = inspect.getframeinfo(ref)
                print(f"      Frame: {frame_info.filename}:{frame_info.lineno}")
                
                # Show local variables in the frame
                if hasattr(ref, "f_locals"):
                    locals_keys = list(ref.f_locals.keys())[:5]  # First 5 local variables
                    print(f"      Frame locals: {locals_keys}")
