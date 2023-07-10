from pynput import keyboard
import threading

class KeyboardListener(threading.Thread):
    """
        Customized keyboard listener
    """
    def __init__(self):
        threading.Thread.__init__(self)
        self._pause = False
        self._left = 0
        self._esc = False
        self._terminated = False
        self._save = False
        self._pcd = True
        self._model = True
        self._ft = False

    def press_action(self, key):
        if self._terminated or key == keyboard.Key.esc:
            self._esc = True
            return False
        if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r: self._pause ^= True
        elif key == keyboard.Key.left: self._left += 1
        elif key == keyboard.Key.right: self._left -= 1
        elif hasattr(key, "char"):
            if key.char == 'c' or key.char == 'C': self._save = True
            elif key.char == 'p' or key.char == 'P': self._pcd = not self._pcd
            elif key.char == 'm' or key.char == 'M': self._model = not self._model
            elif key.char == 'f' or key.char == 'F': self._ft = not self._ft
            
    def release_action(self, key): pass

    def terminate(self): self._terminated = True

    def run(self):
        with keyboard.Listener(on_press=self.press_action, on_release=self.release_action) as listener: listener.join()
            
    @property
    def pause(self): return self._pause
    
    @property
    def left(self): return self._left
    @left.setter
    def left(self, val:int): self._left = val

    @property
    def save(self): return self._save
    @save.setter
    def save(self, val:bool): self._save = val
    
    @property
    def esc(self): return self._esc
    
            
    