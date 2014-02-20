__author__ = 'Lothilius'

class MyQueue:
    """Will take items and place them in to a list so that a FIFO structure is used."""
    def __init__(self):
        self._items = []

    def __str__(self):
        output = ""
        for x in self._items:
            output = output + str(x) + " "
        return output

    def enqueue(self, item):
        self._items.insert(0, item)

    def dequeue(self):
        if MyQueue.isEmpty(self):
            return "The Stack is empty."
        else:
            return self._items.pop()

    def __len__(self):
        return len(self._items)

    def isEmpty(self):
        return not len(self._items)

    def peek(self):
        return self._items[-1]

class Stack:
    """Defines a Stack ADT with operations: Stack, isEmpty,
        push, pop, peek, and len."""
    def __init__(self):
        self._items = []

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __str__(self):
        output = ""
        for x in self._items:
            output = str(x) + " " + output
        return "[ " + output + "]"

    def isEmpty(self):
        return not len(self._items)

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if Stack.isEmpty(self):
            return "The Stack is empty."
        else:
            return self._items.pop()

    def peek(self):
        return self._items[-1]

    def next(self):
        if len(self._items) > 0:
            return self._items.pop()
        else:
            raise StopIteration