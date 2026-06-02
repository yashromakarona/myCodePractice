class Node(object):
    def __init__(self, data=None):
        self._data = data
        self._next = None  
    
    @property
    def data(self):
        return self._data
    
    @property
    def next(self):
        return self._next
    
    @data.setter
    def data(self, data):
        self._data = data

    @next.setter
    def next(self, next):
        self._next = next


class SinglyLinkedList(object):
    def __init__(self):
        self._head = None
        self._tail = None
        self._num_nodes = 0

    def __len__(self):
        return self._num_nodes

    def empty(self):
        if self._num_nodes == 0:
            return True
        else:
            return False

    def insert(self, i, data):
        if i < 0 or i > self._num_nodes:
            raise IndexError

        new_node = Node(data)

        if self._num_nodes == 0:
            self._head = self._tail = new_node
        elif i == 0:
            new_node.next = self._head
            self._head = new_node
        elif i == self._num_nodes:
            self._tail.next = new_node
            self._tail = new_node
        else:
            curr = self._head
            for _ in range(i - 1):
                curr = curr.next
            new_node.next = curr.next
            curr.next = new_node

        self._num_nodes += 1
        
    def remove(self, i):
        pass

    def clear(self):
        pass

    def get(self, i):
        pass

    def pop(self, i=None):
        pass
    
    def search(self, target, start=0):
        pass

    def extend(self, sll):
        pass

node = Node()
node2 = Node()
node3 = Node()
node4 = Node()

node.data = 1
node.next = node2

node2.data = 2
node2.next = node3

node3.data = 3
node3.next = node4

node4.data = 4
node4.next = None


print(node.data)
print(node.next.data)
print(node.next.next.data)
print(node.next.next.next.data)