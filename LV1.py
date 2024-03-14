'''
#zadatak 1
def total_euro(sati, satnica) :
    return float(sati) * float(satnica)

print("Radni sati:")
sati = input()
print("eura/h:")
satnica = input()
print("Radni sati: ",sati, "h", "\neura/h: ", satnica, "\n3Ukupno: ", float(sati)*float(satnica), "eura")
total = total_euro(sati, satnica)
print(total)
'''
'''
#zadatak 2
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
print("Upišite vrijednost iz intervala [0.0, 1.0].")
x = float(input())
if(isfloat(x)) :
    if(0.0 <= x <= 1.0) :
        if(x >= 0.9) :
            print("A")
        elif(x >= 0.8) :
            print("B")
        elif(x >= 0.7) :
            print("C") 
        elif(x >= 0.6) :
            print("D")
        else :
            print("F")
    else :
        print("Upisani broj se ne nalazi u intervalu.")
else :
    print("Upisana vrijednost nije broj.")
'''
'''
#zadatak 3
list = []
while True:
    unos = input()
    if(unos.lower() == 'done'):
        break
    if isfloat(unos):
        list.append(float(unos))
    else:
        print("Nije broj.")
print("Broj znamenki", len(list))
print("Srednja vrijednost", sum(list)/len(list))
print("Maksimalna vrijednost", max(list))
print("Minimalna vrijednost", min(list))
list.sort()
print(list)
'''
'''
#zadatak 4
rijecnik = {}
fhand = open("C:/Users/student/Desktop/lv1/song.txt")
for line in fhand:
    line = line.rstrip()
    words = line.split()
    for word in words:
        rijec = word.lower()
        if rijec in rijecnik:
            rijecnik[rijec] += 1
        else:
            rijecnik[rijec] = 1
fhand.close()
broj = 0
for rj in rijecnik:
    if(rijecnik[rj]==1):
        broj +=1
print(broj)
'''
#zadatak 5
with open("C:/Users/student/Desktop/lv1/SMSSpamCollection.txt", "r") as file:
    sms_data = file.readlines()
ham_count = 0
ham_wordcount = 0
spam_count = 0
spam_wordcount = 0
counter = 0
for line in sms_data:
    label, message = line.split(None, 1)
    if(label == 'ham'):
        ham_count += 1
        ham_wordcount += len(message.split())
    elif(label == 'spam'):
        spam_count += 1
        spam_wordcount += len(message.split())
        if line.endswith('!'):
            counter += 1
print(counter)
if(ham_count > 0):
    print("Prosječan broj ham poruka", int(ham_wordcount/ham_count))
else:
    print("Nema ham poruka")
if(spam_count > 0):
    print("Prosječan broj spam poruka", int(spam_wordcount/spam_count))
else:
    print("Nema spam poruka")