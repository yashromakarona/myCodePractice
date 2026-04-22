class History:
    
    def __init__(self):
        self._stack_undo = []
        self._stack_redo = []
        
    def append(self, state):
        self._stack_undo.append(state)
        if len(self._stack_redo) != 0:
            self._stack_redo.clear()
    
    def undo(self):
        if len(self._stack_undo) == 0:
            return
        
        item = self._stack_undo.pop()
        self._stack_redo.append(item)
        
    def redo(self):
        if len(self._stack_redo) == 0:
            return
        
        item = self._stack_redo.pop()
        self._stack_undo.append(item)
        
        
history = History()
history.append(1)
history.append(2)
history.append(3)

history.undo()
history.undo()
history.redo()

print("[UNDO] : ", history._stack_undo)
print("[REDO] : ", history._stack_redo)