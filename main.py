import math
import random

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
file_path=""
# Функция для выбора файла
dictionary={}
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        label_file.config(text=f"Файл: {file_path}")
        try:
            with open(file_path, encoding='utf-8') as file:
                for line in file:
                    words = line.split()
                    if len(words) == 2:
                        try:
                            key = words[0]
                            value = float(words[1].replace(',', '.'))  # Преобразуем строку в число
                            dictionary[key] = value  # Сохраняем биграмму в словарь
                        except ValueError:
                            print(f"Ошибка преобразования: {words[1]} в строке: {line}")
                    else:
                        print(f"Некорректная строка: {line}")
            print("Биграммы успешно загружены в словарь!")
        except Exception as e:
            print(f"Ошибка при открытии файла: {e}")
arr=[]
numbers_coord=[]
dictionary={}
dlina=1
matrix=[]


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




def find_all_element(matrix1, target):
    coordinates = []
    for i in range(len(matrix1)):
        for j in range(len(matrix1[i])):
            if matrix1[i][j] == target:
                coordinates.append((i, j))
    return coordinates



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

def run_algorithm():
    global P_CROSSOVER
    global POPULATION_SIZE
    global P_MUTATION
    global MAX_GENERATIONS
    global start_text
    global arr
    global  dlina,numbers_coord,dictionary,matrix
    output_field.delete(1.0, tk.END)
    try:
        POPULATION_SIZE = int(entry_population_size.get())
        P_CROSSOVER = float(entry_p_crossover.get())
        P_MUTATION = float(entry_p_mutation.get())
        MAX_GENERATIONS = int(entry_max_generations.get())
        start_text = input_field.get(1.0, tk.END).strip()

        # Здесь должна быть логика вашего генетического алгоритма с использованием input_data
        output_text = (
            f"Алгоритм запущен с параметрами:\n"
            f"POPULATION_SIZE = {POPULATION_SIZE}\n"
            f"P_CROSSOVER = {P_CROSSOVER}\n"
            f"P_MUTATION = {P_MUTATION}\n"
            f"MAX_GENERATIONS = {MAX_GENERATIONS}\n"
            f"Входные данные:\n{start_text}\n"
            "Запуск генетического алгоритма...\n"
            "Результаты будут здесь."
        )

        # Вывод результата в текстовое поле
        output_field.delete(1.0, tk.END)
        output_field.insert(tk.INSERT, output_text)

    except ValueError:
        output_field.delete(1.0, tk.END)
        output_field.insert(tk.INSERT, "Ошибка ввода! Убедитесь, что ввели корректные значения.")
    start_text = start_text
    dlina = math.sqrt(len(start_text))
    print(dlina)
    right = False
    if dlina.is_integer():
        print("Результат логарифма является целым числом")
        if (dlina % 2 == 0):
            right = True
    else:
        print("Результат логарифма не является целым числом")
    n = int(dlina)
    arr = [[0] * int(n / 2) for _ in range(int(n / 2))]
    t = 1
    for i in range(int(n / 2)):
        for j in range(int(n / 2)):
            arr[i][j] = t
            t += 1
    # arr[i]+=reversed(arr[i])
    # arr+=reversed(arr)
    Second_Part = []
    Next_part = (rotate_matrix(arr))
    print(arr)
    print(Next_part)
    for i in range(len(arr)):
        arr[i] += Next_part[i]
        Second_Part.insert(0, list(reversed(arr[i])))
    arr += Second_Part



    numbers_coord = []
    for i in range(1, int((dlina / 2) * (dlina / 2) + 1)):
        numbers_coord.append(find_all_element(arr, i))

    matrix = convert_to_matrix(start_text)
    for row in arr:
        print(' '.join(map(str, row)))

    matrixs = []
    for i in range(4):
        matrixs.append(matrix)

        matrix = rotate_matrix(matrix)



    print(dictionary)


    import random

    # константы задачи
    ONE_MAX_LENGTH = int(dlina)  # длина подлежащей оптимизации битовой строки


    population = populationCreator(n=POPULATION_SIZE)
    generationCounter = 0

    fitnessValues = list(map(oneMaxFitness, population))

    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    maxFitnessValues = []
    meanFitnessValues = []

    fitnessValues = [individual.fitness.values[0] for individual in population]
    best = []

    while True:

        print("max=" + str(min(fitnessValues)))
        generationCounter += 1
        offspring = selTournament(population, len(population))
        offspring = list(map(clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                cxOnePoint(child1, child2)

        for mutant in offspring:
            if random.random() < P_MUTATION:
                mutant = mutFlipBit(mutant, indpb=1.0 / ONE_MAX_LENGTH)

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

        output_field.insert(tk.INSERT, (
            f"\nПоколение {generationCounter}: Макс приспособ. = {maxFitness}, Средняя приспособ.= {meanFitness}"))
        output_field.see(tk.END)
        output_field.update()

        best_index = fitnessValues.index(min(fitnessValues))
        best = population[best_index]

        # Проверка условия выхода
        if min(fitnessValues) <= 0.1 or generationCounter >= MAX_GENERATIONS:
            break



    Hob = ""

    for i in range(4):
        for j in best:
            # Извлекаем символы из матрицы по текущим координатам
            Hob += matrix[j[0]][j[1]]

        # Поворачиваем координаты на 90 градусов для следующей итерации
        best = sorted(rotate_coords(best, len(matrix)))  # 4 - размер матрицы
    output_field.insert(tk.INSERT,("\n"+Hob))
    print(Hob)
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Поколение')
    plt.ylabel('Макс/средняя приспособленность')
    plt.title('Зависимость максимальной и средней приспособленности от поколения')
    plt.show()


# Создание окна
window = tk.Tk()
window.title("Генетический Алгоритм")
window.geometry("800x600")

# Адаптация элементов при изменении размера окна
window.grid_columnconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=1)

# Поля ввода для параметров
label_population_size = tk.Label(window, text="POPULATION_SIZE:")
label_population_size.grid(row=0, column=0, padx=10, pady=5, sticky='w')
entry_population_size = tk.Entry(window, width=20)
entry_population_size.insert(tk.END, "1500")
entry_population_size.grid(row=0, column=1, padx=10, pady=5)

label_p_crossover = tk.Label(window, text="P_CROSSOVER:")
label_p_crossover.grid(row=1, column=0, padx=10, pady=5, sticky='w')
entry_p_crossover = tk.Entry(window, width=20)
entry_p_crossover.insert(tk.END, "0.4")
entry_p_crossover.grid(row=1, column=1, padx=10, pady=5)

label_p_mutation = tk.Label(window, text="P_MUTATION:")
label_p_mutation.grid(row=2, column=0, padx=10, pady=5, sticky='w')
entry_p_mutation = tk.Entry(window, width=20)
entry_p_mutation.insert(tk.END, "0.5")
entry_p_mutation.grid(row=2, column=1, padx=10, pady=5)

label_max_generations = tk.Label(window, text="MAX_GENERATIONS:")
label_max_generations.grid(row=3, column=0, padx=10, pady=5, sticky='w')
entry_max_generations = tk.Entry(window, width=20)
entry_max_generations.insert(tk.END, "2000")
entry_max_generations.grid(row=3, column=1, padx=10, pady=5)

# Поле для большого ввода данных
label_input = tk.Label(window, text="Ввод данных:")
label_input.grid(row=4, column=0, padx=10, pady=5, sticky='w')
input_field = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=10)
input_field.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

# Кнопка для выбора файла
button_file = tk.Button(window, text="Выбрать файл", command=select_file)
button_file.grid(row=6, column=0, pady=10, padx=10, sticky='w')

# Место для отображения выбранного пути файла
label_file = tk.Label(window, text="Файл не выбран")
label_file.grid(row=6, column=1, pady=10, sticky='w')

# Кнопка для запуска алгоритма
start_button = tk.Button(window, text="Начать алгоритм", command=run_algorithm)
start_button.grid(row=7, column=0, columnspan=2, pady=10)

# Большое поле вывода
output_field = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=10)
output_field.grid(row=8, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

# Запуск основного цикла приложения
window.mainloop()


