class History:
    
    def __init__(self):
        self._stack_undo = []
        self._stack_redo = []
        
    def append(self, state):
        self._stack_undo.append(state)
        if len(self._stack_redo) != 0:
            self._stack_redo = []
            
    def undo(self):
        if len(self._stack_undo) == 0:
            return None
        else:
            self._stack_redo.append(self._stack_undo.pop())
            return self._stack_redo[-1]
            
    def redo(self):
        if len(self._stack_redo) == 0:
            return None
        else:
            self._stack_undo.append(self._stack_redo.pop())
            return self._stack_undo[-1]
            
    def get_undo_stack(self):
        return self._stack_undo
        
    def get_redo_stack(self):
        return self._stack_redo