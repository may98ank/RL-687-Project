import numpy as np 

class CacheEnv:
    def __init__(self, num_pages, cache_size, episode_len):
        self.num_pages = num_pages
        self.cache_size = cache_size
        self.episode_len = episode_len
        
        self.t = 0 # time step
        self.current_request = None
        self.cache = None # list of page indices in the cache (not binary)

    def _sample_request(self):
        return np.random.randint(0, self.num_pages)
    
    def _get_state(self):
        cache_vec = np.zeros(self.num_pages)
        for page in self.cache:
            cache_vec[page] = 1
        req_vec = np.zeros(self.num_pages)
        req_vec[self.current_request] = 1
        state = np.concatenate([cache_vec, req_vec], axis=0)
        return state

    def reset(self):
        self.t = 0
        self.current_request = self._sample_request() # sample the first request
        self.cache = []  # initial empty cache
        return self._get_state()
    
    def print_cache(self):
        """Print the pages in the cache in a readable format."""
        print(f"Cache: {self.cache}")
    
    def print_state(self):
        """Print the current state in a readable format."""
        cache_vec = np.zeros(self.num_pages)
        for page in self.cache:
            cache_vec[page] = 1
        req_vec = np.zeros(self.num_pages)
        req_vec[self.current_request] = 1
        
        print(f"\n{'='*60}")
        print(f"Time Step: {self.t}")
        print(f"{'='*60}")
        print(f"Cache contents: {sorted(self.cache) if self.cache else '[]'}")
        print(f"Cache binary:    {cache_vec.astype(int)}")
        print(f"Current request:  Page {self.current_request}")
        print(f"Request binary:   {req_vec.astype(int)}")
        print(f"Cache size: {len(self.cache)}/{self.cache_size}")
    
    def print_step_info(self, action, reward, hit, evicted, cache_before, cache_after, request):
        """Print detailed information about a step."""
        print(f"\n{'─'*60}")
        print(f"Step {self.t}:")
        print(f"  Cache (before): {cache_before if cache_before else '[]'}")
        print(f"  Request:        Page {request}")
        print(f"  Action:         Evict cache slot {action}")
        if cache_before:
            print(f"  Cache slot {action}: Page {cache_before[action] if action < len(cache_before) else 'N/A'}")
        print(f"  Result:         {'HIT ✓' if hit else 'MISS ✗'}")
        if not hit:
            if len(cache_before) < self.cache_size:
                print(f"  Operation:      Cache not full → Insert page {request} (action ignored)")
            else:
                if evicted is not None:
                    print(f"  Operation:      Evicted page {evicted} from slot {action} → Insert page {request}")
                else:
                    print(f"  Operation:      Evicted from slot {action} → Insert page {request}")
        else:
            print(f"  Operation:      No change (page already in cache)")
        print(f"  Reward:         {reward:+d}")
        print(f"  Cache (after):  {cache_after if cache_after else '[]'}")
    def step(self, action, verbose=False):
        """
        Execute one step in the environment.
        
        Args:
            action: Cache slot index (0 to cache_size-1) to evict from (if eviction is needed)
            verbose: If True, print detailed step information
        
        Returns:
            next_state, reward, done, info
        """
        # Store cache state before update for printing
        cache_before = self.cache.copy()
        request_before = self.current_request
        
        hit = self.current_request in self.cache
        reward = 1 if hit else -1
        evicted = None

        if not hit:
            if len(self.cache) < self.cache_size:  # cache is not full 
                # still room in the cache, so just add the new page
                # Action is ignored when cache is not full
                self.cache.append(self.current_request)
            else:
                # Cache is full, action is a slot index (0 to cache_size-1)
                # Clamp action to valid range
                action = min(action, len(self.cache) - 1)
                evicted = self.cache[action]
                self.cache.pop(action)  # Remove page at slot 'action'
                self.cache.append(self.current_request)

        cache_after = self.cache.copy()
        
        if verbose:
            self.print_step_info(action, reward, hit, evicted, cache_before, cache_after, request_before)
        
        self.t += 1
        done = self.t >= self.episode_len  # episode is done if the time step is greater than the episode length
        self.current_request = self._sample_request()  # sample the next request
        return self._get_state(), reward, done, {"hit": hit, "evicted": evicted}


# smoke test the environment
if __name__ == "__main__":
    env = CacheEnv(num_pages=10, cache_size=3, episode_len=10)
    
    print("="*60)
    print("CACHE REPLACEMENT ENVIRONMENT TEST")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Number of pages: {env.num_pages}")
    print(f"  - Cache size: {env.cache_size}")
    print(f"  - Episode length: {env.episode_len}")
    print("="*60)
    
    state = env.reset()
    print("\n🔵 INITIAL STATE:")
    env.print_state()
    
    total_reward = 0
    states = []
    actions = []
    rewards = []
    requests = []
    hits = []
    cache_states = []
    
    print("\n" + "="*60)
    print("EPISODE EXECUTION")
    print("="*60)
    
    for t in range(env.episode_len):
        # Store current request before step (since step updates it)
        current_req = env.current_request
        action = np.random.randint(0, env.cache_size)
        next_state, reward, done, info = env.step(action, verbose=True)
        total_reward += reward
        state = next_state
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        requests.append(current_req)
        hits.append(info["hit"])
        cache_states.append(sorted(env.cache.copy()))  # Store cache state after step
        if done:
            break
    
    print("\n" + "="*60)
    print("EPISODE SUMMARY")
    print("="*60)
    print(f"\n📊 Statistics:")
    num_hits = sum(1 for r in rewards if r == 1)
    num_misses = len(rewards) - num_hits
    hit_rate = num_hits / len(rewards) if len(rewards) > 0 else 0.0
    print(f"  Total steps: {len(rewards)}")
    print(f"  Hits: {num_hits} ({hit_rate*100:.1f}%)")
    print(f"  Misses: {num_misses} ({(1-hit_rate)*100:.1f}%)")
    print(f"  Total reward: {total_reward}")
    print(f"  Average reward: {np.mean(rewards):.2f}")
    
    print(f"\n📋 Step-by-step breakdown:")
    print(f"{'Step':<6} {'Request':<8} {'Action':<8} {'Hit?':<6} {'Reward':<8} {'Cache After'}")
    print("-" * 70)
    for i in range(len(rewards)):
        cache_after = cache_states[i] if i < len(cache_states) else []
        hit_str = "✓" if hits[i] else "✗"
        cache_str = str(cache_after) if cache_after else "[]"
        print(f"{i+1:<6} {requests[i]:<8} {actions[i]:<8} {hit_str:<6} {rewards[i]:+8} {cache_str}")
    
    print("\n" + "="*60)