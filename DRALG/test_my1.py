class History(object):
    def __init__(self):
        self._stack_undo = []
        self._stack_redo = []

    def current_state(self):
        n = len(self._stack_undo)

        if len(self._stack_undo) != 0:    
            print(self._stack_undo[n-1])
        else:
            return None

    def append(self, state):
        if len(self._stack_redo) != 0:
            self._stack_redo.clear()
        self._stack_undo.append(state)

    def undo(self):
        if len(self._stack_undo) == 0:
            return
        else:
            item = self._stack_undo.pop()
            self._stack_redo.append(item)

    def redo(self):
        if len(self._stack_redo) == 0:
            return
        else:
            item = self._stack_redo.pop()
            self._stack_undo.append(item)

    def clear(self):
        self._stack_undo.clear()
        self._stack_redo.clear()

history = History()
history.append(1)
history.append(2)
history.append(3)

history.undo()
history.undo()
history.redo()

print("[UNDO] : ", history._stack_undo)
print("[REDO] : ", history._stack_redo)