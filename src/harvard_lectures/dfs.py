import termcolor

class Node():
    def __init__(self, x, y, parent):
        self.x = x
        self.y = y
        self.parent = parent

class Stack():
    def __init__(self):
        self.visited_nodes = []
    
    def empty(self):
        return len(self.visited_nodes) == 0

    def any(self, node):
        return any(cnode.x == node.x and cnode.y == node.y for cnode in self.visited_nodes) 

    def add(self, node):
        self.visited_nodes.append(node)

    def remove(self):
        if self.empty():
            raise Exception(f'cannot remove from empty stack')

        top = self.visited_nodes[-1]
        self.visited_nodes = self.visited_nodes[:-1]
        return top

class Queue(Stack):
    def remove(self): #override
        if self.empty():
            raise Exception(f'cannot remove from empty queue')
        top = self.visited_nodes[0]
        self.visited_nodes = self.visited_nodes[1:]
        return top

class Maze():

    def __init__(self, goal_node: Node):
        self.goal_node = goal_node
        self.visited = Stack() # use only for the overridden any I built
        
    def _create_maze(self, start_node: Node):
        self.maze_internal = [2,2]
        self.current_node = start_node
        self.maze_internal = [
            [start_node, 'x', 'x'],
            ['o', 'o', 'o'],
            ['x', 'x', 'o']
        ]

    def solve(self, current_node: Node):
        if (current_node.x > 2 or current_node.x < 0 or current_node.y > 2 or current_node.y < 0):
            termcolor.cprint('You are out of bounds!', "yellow")
            return
        
        if (self.visited.any(self.goal_node)):
            return
        
        if (self.is_goal_node(current_node)):
            termcolor.cprint('Solved!', "green")
            self.visited.add(current_node)
            return

        if (self.maze_internal[current_node.x][current_node.y] == 'x'):
            termcolor.cprint('Move not allowed. Backtracking.', "red")
            return
        else:
            self.visited.add(current_node)
            print(f'[{current_node.x},{current_node.y}] Added to the path.')
            current_node = Node(current_node.x+1,  current_node.y, current_node)
            self.solve(current_node)
            
            if not self.visited.any(self.goal_node):
                self.visited.remove()
                current_node = Node(current_node.x, current_node.y+1, current_node)
                self.solve(current_node)
          
    def is_goal_node(self, node: Node):
        return node.x == self.goal_node.x and node.y == self.goal_node.y
        
class Main:
    @staticmethod
    def Execute():
        maze = Maze(Node(2,2, None))
        start_node = Node(0,0, None)
        maze._create_maze(start_node)
        maze.solve(start_node)

if __name__ == "__main__":
    Main.Execute()