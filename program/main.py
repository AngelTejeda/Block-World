from __future__ import annotations
import copy
import time
import problems
import csv

class Move:
    def __init__(self, block: str, stack: int, over: str):
        self.block = block
        self.stack = stack
        self.over = over

    def __repr__(self):
        string = f"- Move block {self.block} "

        if self.over != None:
            string += f"to stack {self.stack} over block {self.over}."
        else:
            string += "to a new stack."

        return string


class State:
    def __init__(self, stacks: list):
        blocks: set = set()

        # Quita los stacks vacíos.
        while True:
            try:
                stacks.remove([])
            except:
                break

        # Verifica los bloques de los stacks.
        for stack in stacks:
            if type(stack) is not list:
                raise Exception(f"{str(stack)} is not a list.")

            for block in stack:
                if type(block) is not str:
                    raise Exception(f"{str(block)} is not a valid block name.")

                if block in blocks:
                    raise Exception(f"The block {block} is duplicated")

                blocks.add(block)

        self.stacks = stacks

    def __repr__(self):
        stacks: str = ""

        for i in range(len(self.stacks)):
            stacks += f"\tStack {i+1}: {self.stacks[i]}\n"

        return stacks

    def __eq__(self, other: State):
        for stack in self.stacks:
            
            found = False
            
            for stack_2 in other.stacks:
                if stack == stack_2:
                    found = True
                    break

            if not found:
                return False

        return True
    
    # Cuenta el número de bloques que hay en un estado.
    def count_blocks(self):
        cont = 0
        for i in range(len(self.stacks)):
            for j in range(len(self.stacks[i])):
                cont += 1
        
        return cont

    # Aplica un movimiento al estado actual y devuelve el estado resultado.
    def move(self, from_stack: int, to_stack: int) -> State:
        # Si el bloque está sobre la mesa (es el único bloque en el stack), no 
        # puede moverse a un nuevo stack.
        if len(self.stacks[from_stack]) == 1 and to_stack > len(self.stacks):
            return None

        new_state: State = copy.deepcopy(self)
        stacks: list = new_state.stacks

        top: str = stacks[from_stack].pop()

        # Si 'to_stack' corresponde a un stack que no existe, uno nuevo se crea.
        if to_stack == len(self.stacks):
            stacks.append([])
        
        stacks[to_stack].append(top)

        # Si 'from_stak' queda vacío luego del movimiento, se elimina.
        if stacks[from_stack] == []:
            stacks.pop(from_stack)

        return new_state


class Node:
    def __init__(self, state: State):
        self.state: State = state
        self.h = 0
        self.g = 0
        self.f = 0
        self.parent = None
        self.move: Move = None
        self.children = []

    def __repr__(self):
        return f"{str(self.move)}\n{self.state}"

    def __eq__(self, other):
        return self.state == other.state

    # Heuristica 1
    # Cuenta el número de bloques en la posición erronea.
    def funcion_heuristica(self, goal_state: State) -> None:
        stacks: list = self.state.stacks

        # Recorre las stacks del nodo actual.
        for i in range(len(stacks)):
            # Recorre los elementos del stack.
            for j in range(len(stacks[i])):
                correct_position: bool = False
                
                # Busca si algún stack del estado meta tiene al elemento que se
                # está comparando en la misma posición.
                for k in range(len(goal_state.stacks)):
                    try:
                        if goal_state.stacks[k][j] == stacks[i][j]:
                            correct_position = True
                            break
                    except:
                        continue

                self.h += 0 if correct_position else 1

    # Calcula los valores de g, h y f y los guarda.
    def evaluate(self, goal_state: State) -> None:
        # Calcular g
        if self.parent:
            self.g = self.parent.g + 1
        
        # Calcular h
        self.funcion_heuristica(goal_state)
                            
        # Calcular f
        self.f = self.g + self.h

    # Genera los hijos del nodo.
    def explore(self, goal_state: State, max_stacks) -> None:
        total_stacks: int = len(self.state.stacks)

        for i in range(total_stacks):
            # Bloque que está siendo movido
            block: str = self.state.stacks[i][-1]

            for j in range(total_stacks + 1):
                if i == j:
                    continue

                # Si se llega al límite de stacks y se intenta mover un block
                # a una nueva stack
                if j == total_stacks and len(self.state.stacks) >= max_stacks:
                    continue

                # Bloque sobre el que se pondrá.
                over: str = None if j == total_stacks else self.state.stacks[j][-1]

                new_state: State = self.state.move(i, j)

                if new_state is None:
                    continue

                # Valores del hijo.
                child: Node = Node(new_state)

                child.parent = self
                child.evaluate(goal_state)
                child.move = Move(block, j+1, over)

                self.children.append(child)


# Cola de prioridad utilizada en A*.
class PriorityQueue():
    def __init__(self):
        self.queue = []

    def __repr__(self):
        return ' '.join([i.state.stacks for i in self.queue])

    def __len__(self):
        return len(self.queue)

    # Recibe un nodo. Si en la cola hay un nodo con el mismo estado, devuelve
    # el nodo de la cola.
    def get_node_in_queue(self, node: Node):
        for elem in self.queue:
            if elem == node:
                return elem

        return None

    def is_empty(self):
        return len(self) == 0

    # Agrega un nodo a la cola.
    def push(self, node: Node):
        self.queue.append(node)

    # Elimina el nodo con el valor mínimode f en la cola y lo devuelve.
    def pop_minimum(self):
        if self.is_empty():
            return None

        min_cost: int = self.queue[0].f
        min_pos: int = 0

        for i in range(len(self.queue)):
            curr_node: Node = self.queue[i]

            if curr_node.f < min_cost:
                min_cost = curr_node.f
                min_pos = i

        return self.queue.pop(min_pos)


# Clase auxiliar para generar los reportes.
class ReportValues:
    def __init__(self, instance, configuration, heuristic, blocks, stack_limit):
        self.max_size = 1
        self.curr_size = 1
        self.movements = 1
        self.total_nodes = 0
        self.time = 0
        self.cost = float('inf')
        
        self.instance = instance
        self.configuration = configuration
        self.heuristic = heuristic
        self.blocks = blocks
        self.stack_limit = stack_limit
        
    def __repr__(self):
        string = f"Max PQ Size: {self.max_size}\n"
        string += f"PQ Operations: {self.movements}\n"
        string += f"Total Generated Nodes: {self.total_nodes}\n"
        string += f"Time: {self.time}"
        
        return string

    def poped_element(self):
        self.curr_size -= 1
        self.movements += 1
    
    def appended_element(self):
        self.curr_size += 1
        self.movements += 1
    
    def add_nodes(self, nodes):
        self.total_nodes += nodes
        
    def update_max_size(self):
        if self.curr_size > self.max_size:
            self.max_size = self.curr_size
            
    def writeReportValues(self):
        filename = "report.csv"
        fieldnames = ['instance',
                      'configuration',
                      'blocks',
                      'heuristic',
                      'stack_limit',
                      'max_pq_size',
                      'pq_operations',
                      'total_nodes',
                      'cost',
                      'time']
        
        try:
            open(filename, 'r', newline='')
        except:
            with open(filename, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                
                writer.writeheader()
        
        with open(filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
        
            writer.writerow({'instance': self.instance,
                             'configuration': self.configuration,
                             'blocks': self.blocks,
                             'heuristic': self.heuristic,
                             'stack_limit': self.stack_limit,
                             'max_pq_size': report.max_size,
                             'pq_operations': report.movements,
                             'total_nodes': self.total_nodes,
                             'cost': self.cost,
                             'time': self.time
                             })


# Algoritmo Informado A*
def a_star(
        initial_state: State,
        goal_state: State,
        report: ReportValues,
        max_stacks=float('inf')):
    
    if max_stacks < len(initial_state.stacks):
        raise Exception(
            "max_stacks must be greater than or equal to the number of stacks in the initial state.")

    open_list: PriorityQueue = PriorityQueue() # Nodos que faltan ser explorados.
    closed_list: list = [] # Nodos que han sido explorados.

    # Nodo con el estado incial
    initial_node: Node = Node(initial_state)
    
    # Se evalúa al nodo inicial.
    initial_node.evaluate(goal_state)

    # Se agrega el nodo inicial a la lista de nodos por explorar.
    open_list.push(initial_node)
    
    # Inicio del algoritmo y el temporizador.
    start_time = time.time()
    while not open_list.is_empty() and time.time() - start_time < 60:
        # Se obtiene el nodo con valor mínimo f.
        curr_node = open_list.pop_minimum()

        report.poped_element()  # Reporte, no relacionado al algoritmo.

        if curr_node.state == goal_state:
            end_time = time.time()
            
            report.time = end_time - start_time # Reporte, no relacionado al algoritmo.
            report.cost = curr_node.f # Reporte, no relacionado al algoritmo.
            
            return curr_node
        
        # Se explora el nodo y se añade a la lista de explorados.
        closed_list.append(curr_node.state)
        curr_node.explore(goal_state, max_stacks)
        
        report.add_nodes(len(curr_node.children)) # Reporte, no relacionado al algoritmo.

        # Los hijos del nodo explorado se analizan.
        for child in curr_node.children:
            if child.state in closed_list:
                continue

            open_list_node = open_list.get_node_in_queue(child)

            # Si el nodo no está en la lista abierta, o si está pero tiene un
            # costo menor que el que existe.
            if open_list_node is None or child.g <= open_list_node.g:
                open_list.push(child)
                
                report.appended_element() # Reporte, no relacionado al algoritmo.

            report.update_max_size() # Reporte, no relacionado al algoritmo.
    
    
    end_time = time.time()
    
    report.time = end_time - start_time # Reporte, no relacionado al algoritmo.
    
    return None


# Algoritmo Desinformado DFS
def dfs(
        initial_state: State,
        goal_state: State,
        report,
        max_stacks=float('inf')):
    
    if max_stacks < len(initial_state.stacks):
        raise Exception(
            "max_stacks must be greater than or equal to the number of stacks in the initial state.")
    
    visited = [] # Nodos que han sido visitados.
    stack = [] # Nodos que no han sido expandidos.
    
    # Nodo con el estado inicial.
    initial_node = Node(initial_state)
    
    # Se agrega el nodo inicial a la lista de nodos por explorar.
    stack.append(initial_node)
    
    # Inicia el contador y el algoritmo.
    start_time = time.time()
    while len(stack) and time.time() - start_time < 60:
        # Se obtiene el nodo en el tope del stack.
        node = stack.pop()
        
        report.poped_element() # Reporte, no relacionado al algoritmo.
        
        # Si el nodo es un nodo meta.
        if node.state == goal_state:
            end_time = time.time()
            
            report.time = end_time - start_time # Reporte, no relacionado al algoritmo.
            report.cost = node.f # Reporte, no relacionado al algoritmo.
            return node
        
        # Si el nodo no ha sido visitado.
        if node not in visited:
            visited.append(node)
        
        # Se explora el nodo
        node.explore(goal_state, max_stacks)
        
        report.add_nodes(len(node.children)) # Reporte, no relacionado al algoritmo.
        
        # Se analizan los hijos del nodo.
        for child in node.children:
            # Si el nodo no ha sido visitado, se agrega al stack.
            if child not in visited:
                stack.append(child)
                
                report.appended_element() # Reporte, no relacionado al algoritmo.
            
            report.update_max_size() # Reporte, no relacionado al algoritmo.
    
    
    end_time = time.time()
    
    report.time = end_time - start_time # Reporte, no relacionado al algoritmo.
    
    return None


# Imprime la serie de pasos que se debe llevar a cabo para llegar del estado
# inical al nodo solución encontrado.
def print_solution(solution_node: Node):
    if solution_node is None:
        print("No solution was found.")
        return

    print(f"A solution with cost {solution_node.g} was found.")

    steps = ""

    current_node = solution_node
    while True:

        if current_node.parent is None:
            break

        steps = "\n" + str(current_node) + steps
        current_node = current_node.parent

    print("Initial State:\n", current_node.state)
    print("Goal State:\n", solution_node.state)

    print(steps)



##########################################


problem_configuration = problems.problems[3]
start: list = problem_configuration[0]
goal: list = problem_configuration[1]
stacks = float('inf')
#stacks = 3


initial_state: State = State(start)
goal_state: State = State(goal)

report = ReportValues(1, 1, 1, initial_state.count_blocks(), stacks)
s = dfs(initial_state, goal_state, report, stacks)
print("F" if s is None else s.f)
print_solution(s)
print(report)

print("-----")

report = ReportValues(1, 1, 1, initial_state.count_blocks(), stacks)
nodo_solucion = a_star(initial_state, goal_state, report, stacks)
print("F" if nodo_solucion is None else nodo_solucion.f)
print_solution(nodo_solucion)
print(report)
#report.writeReportValues()
