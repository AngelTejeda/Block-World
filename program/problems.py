problems = []

# Configuración 0
i0 = [
    ['A', 'B'],
    ['C', 'D']
]

f0 = [
    ['C', 'B'],
    ['A', 'D'],
]

# Configuración 1
i1 = [
      ['D'],
      ['C', 'B', 'A']
]

f1 = [
      ['C', 'D', 'B', 'A']
]

# Configuración 2
i2 = [
      ['A', 'B'],
      ['D', 'C']
]

f2 = [
      ['D', 'B', 'C', 'A']
]

# Configuración 3
i3 = [
      ['D', 'E', 'A', 'C', 'B']
]

f3 = [
      ['E', 'D', 'C', 'B', 'A']
]

# Configuración 4
i4 = [
    ['D', 'A'],
    ['E', 'B'],
    ['F', 'C']
]

f4 = [
    ['A', 'D'],
    ['B', 'E'],
    ['C', 'F']
]

# Configuración 5
i5 = [
    ['G', 'D', 'A'],
    ['H', 'E', 'B'],
    ['I', 'F', 'C'],
]

f5 = [
    ['H', 'E', 'B'],
    ['C', 'F', 'I'],
    ['A', 'D', 'G'],
]


problems.append((i0, f0))
problems.append((i1, f1))
problems.append((i2, f2))
problems.append((i3, f3))
problems.append((i4, f4))
problems.append((i5, f5))