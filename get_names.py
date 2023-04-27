names = []

with open('names2.txt', 'r') as file:
    for line in file:
        temp =''
        for i in range(11,len(line)):
            if line[i] == '\n':
                break
            temp+=line[i]
        names.append(temp)

names.remove('communityname string')