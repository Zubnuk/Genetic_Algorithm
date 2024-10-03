import math
start_text = input()
dlina=math.sqrt(len(start_text))
print(dlina)
right=False
if dlina.is_integer():
    print("Результат логарифма является целым числом")
    if(dlina%2==0):
      right=True
else:
    print("Результат логарифма не является целым числом")

def rotate_matrix(matrix):
    n = len(matrix)
    # Создаем новую матрицу, чтобы сохранить повернутую версию
    rotated_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            # Поворот матрицы осуществляется путем обмена строк и столбцов
            rotated_matrix[j][n - i - 1] = matrix[i][j]

    return rotated_matrix
def rotate_coords(holes,n):
    return [(y, n - 1 - x) for x, y in holes]
def convert_to_matrix(text):
    n = math.isqrt(len(text))
    matrix = [text[i:i+n] for i in range(0, len(text), n)]
    return matrix

n = int(dlina)

arr = [[0]*int(n/2) for _ in range(int(n/2))]
t=1
for i in range(int(n/2)):
    for j in range(int(n/2)):
        arr[i][j]=t
        t+=1
  #arr[i]+=reversed(arr[i])
#arr+=reversed(arr)
Second_Part=[]
Next_part=(rotate_matrix(arr))
print(arr)
print(Next_part)
for i in range(len(arr)):
    arr[i]+=Next_part[i]
    Second_Part.insert(0, list(reversed(arr[i])))
arr+=Second_Part
def find_all_element(matrix1, target):
    coordinates = []
    for i in range(len(matrix1)):
        for j in range(len(matrix1[i])):
            if matrix1[i][j] == target:
                coordinates.append((i, j))
    return coordinates
numbers_coord=[]
for i in range(1,int((dlina/2)*(dlina/2)+1)):
    numbers_coord.append(find_all_element(arr,i))

matrix = convert_to_matrix(start_text)
for row in arr:
    print(' '.join(map(str, row)))

matrixs=[]
for i in range(4):
    matrixs.append(matrix)

    matrix=rotate_matrix(matrix)
dictionary = {}
with open("bigrams.txt", encoding='utf-8') as file:
    for line in file:
        words = line.split()
        if len(words) == 2:
            try:
                key = words[0]
                value = float(words[1].replace(',', '.'))
                dictionary[key] = value
            except ValueError:
                print(f"Ошибка преобразования: {words[1]} в строке: {line}")
        else:
            print(f"Некорректная строка: {line}")

print(dictionary)


import random
import matplotlib.pyplot as plt

# константы задачи
ONE_MAX_LENGTH = int(dlina)    # длина подлежащей оптимизации битовой строки

# константы генетического алгоритма
POPULATION_SIZE = 15  # количество индивидуумов в популяции
P_CROSSOVER = 0.4       # вероятность скрещивания
P_MUTATION = 0.5        # вероятность мутации индивидуума
MAX_GENERATIONS = 10    # максимальное количество поколений
import random
import numpy as np
print(matrix)
class FitnessMax():
    def __init__(self):
        self.values = [0]


class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()
def sort_child(child):
    n = len(child)
    numbers = [i for i in range(1, n + 1)]

    for i in child:
        numbers[arr[i[0]][i[1]]-1]=i

    return numbers
def find_element( target):

    coordinates = numbers_coord[target-1][:]

    return coordinates



def remove_duplicates(population):
    unique_individuals = []
    seen = set()

    for individual in population:
        # Преобразуем особь в кортеж, чтобы использовать множество для проверки
        ind_tuple = tuple(individual)
        if ind_tuple not in seen:
            seen.add(ind_tuple)
            unique_individuals.append(individual)
        else:
            # Если это дубликат, заменяем его на новую особь
            new_individual = individualCreator()
            unique_individuals.append(new_individual)

    return unique_individuals




def oneMaxFitness(individual):
    mat=matrix.copy()
    dict1={}
    text_main=""

    for i in range(4):
        for j in individual:
            # Извлекаем символы из матрицы по текущим координатам
            text_main += mat[j[0]][j[1]]

        # Поворачиваем координаты на 90 градусов для следующей итерации
        individual =sorted( rotate_coords(individual, len(mat)))  # 4 - размер матрицы




    sum1=dlina*dlina
    for i in range(1,int(sum1)):

        if text_main[i-1:i+1] in  dict1.keys():
            dict1[text_main[i-1:i+1]]+=1.0/(sum1-1)
        else:
            dict1[text_main[i-1:i+1]]=1.0/(sum1-1)
    dif=0.0
    m=0

    for i in dict1.keys():
        m+=dict1[i]
    #print("u="+str(m))

    for i in dictionary.keys():
        if i in   (dict1.keys()):
            dif+=abs(dictionary[i]-dict1[i])

        else:
            dif+=dictionary[i]
    #print("ty="+str(dif))
    #print(dict1)
    return dif,



def individualCreator():
    n = int(dlina)


    cord = [0]*int((n/2)*(n/2))

    for i in range(len(cord)):
      target_element = i+1
      result = find_element( target_element)
      cord[i]=random.choice(result)


# Сортируем список координат по их значению
    sorted_coordinates = sorted(cord)
    return Individual(sorted_coordinates)

def populationCreator(n = 0):
    return list([individualCreator() for i in range(n)])


population = populationCreator(n=POPULATION_SIZE)
generationCounter = 0

fitnessValues = list(map(oneMaxFitness, population))

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue

maxFitnessValues = []
meanFitnessValues = []

def clone(value):
    ind = Individual(value[:])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind

def selTournament(population, p_len):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len-1), random.randint(0, p_len-1), random.randint(0, p_len-1)

        offspring.append(min([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))

    return offspring


def selRank(population, p_len):
    # Сортируем популяцию по фитнесу
    sorted_population = sorted(population, key=lambda ind: ind.fitness.values[0])

    # Присваиваем ранги
    ranks = list(range(1, len(sorted_population) + 1))

    # Нормализуем вероятности по рангам
    total_rank = sum(ranks)
    selection_probs = [rank / total_rank for rank in ranks]

    # Выборка по рангам
    offspring = []
    for _ in range(p_len):
        selected = random.choices(sorted_population, weights=selection_probs, k=1)[0]
        offspring.append(selected)

    return offspring


def cxOnePoint(child1, child2):
    s = random.randint(2, int(len(child1)))
    childF = sort_child(child1)
    childS = sort_child(child2)

    childF[s:], childS[s:] = childS[s:], childF[s:]
    child1[:] = sorted(childF)  # Изменяем на месте
    child2[:] = sorted(childS)  # Изменяем на месте

def mutFlipBit(mutant, indpb=0.4):
    for indx in range(len(mutant)):
        if random.random() < indpb:
            num= arr[(mutant[indx])[0]][(mutant[indx])[1]]
            num_list=(find_element(num))
            num_list.remove((mutant[indx]))



            mutant[indx] = random.choice(num_list)
            mutant[:] = sorted(mutant)
            return mutant


fitnessValues = [individual.fitness.values[0] for individual in population]
best=[]

while  min(fitnessValues) > 0.2 and generationCounter < MAX_GENERATIONS:
    print("max="+str(min(fitnessValues)))
    generationCounter += 1
    offspring = selTournament(population, len(population))
    offspring = list(map(clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            cxOnePoint(child1, child2)

    for mutant in offspring:
        if random.random() < P_MUTATION:
            mutant=mutFlipBit(mutant, indpb=1.0/ONE_MAX_LENGTH)
    offspring = remove_duplicates(offspring)
    freshFitnessValues = list(map(oneMaxFitness, offspring))
    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.fitness.values = fitnessValue

    population[:] = offspring

    fitnessValues = [ind.fitness.values[0] for ind in population]

    maxFitness = min(fitnessValues)
    meanFitness = sum(fitnessValues) / len(population)
    maxFitnessValues.append(maxFitness)
    meanFitnessValues.append(meanFitness)
    print(f"Поколение {generationCounter}: Макс приспособ. = {maxFitness}, Средняя приспособ.= {meanFitness}")

    best_index = fitnessValues.index(min(fitnessValues))
    best=population[best_index]


plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()



Hob=""


for i in range(4):
    for j in best:
        # Извлекаем символы из матрицы по текущим координатам
        Hob += matrix[j[0]][j[1]]

    # Поворачиваем координаты на 90 градусов для следующей итерации
    best = sorted(rotate_coords(best, len(matrix)))  # 4 - размер матрицы

print(Hob)
