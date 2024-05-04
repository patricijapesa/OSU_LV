#zadatak 1

def total_euro():
    return float(radni_sati)*float(satnica)

print('Unesite broj radnih sati: ')
radni_sati = input()
print('Unesite satnicu: ')
satnica = input()
ukupno = float(radni_sati)*float(satnica)
print('Radni sati: ',radni_sati, 'h', '\neura/h: ', satnica, '\nUkupno: ', ukupno, 'eura')

ukupno_fja = total_euro()
print('Ukupno putem funkcije: ', ukupno_fja, 'eura')


#zadatak 2

def isFloat(broj):
    try:
        float(broj)
        return True
    except ValueError:
        return False

print('Unesite broj čija je vrijednost između 0.0 i 1.0')
x = float(input())
if(isFloat(x)):
    if(0.0 <= x <= 1.0):
        if(x >= 0.9):
            print('A')
        elif(x >= 0.8):
            print('B')
        elif(x >= 0.7):
            print('C')
        elif(x >= 0.6):
            print('D')
        else:
            print('F')
    else:
        print('Upisani broj se ne nalazi u zadanom intervalu')
else:
    print('Upisana vrijednost nije broj')


#zadatak 3

list = []

print('Unosite brojeve koje zelite u listi te na kraju upisite done ako zelite prestati s unosom')
while True:
    unos = input()
    if(unos.lower() == 'done'):
        break
    if(isFloat(unos)):
        list.append(float(unos))
    else:
        print('Unesena vrijednost nije broj')
print('Dužina liste: ', len(list))
print('Srednja vrijednost brojeva u listi: ', sum(list)/len(list))
print('Minimalna vrijednost: ', min(list))
print('Maksimalna vrijednost: ', max(list))

list.sort()
print(list)


#zadatak 4

fhand = open('song.txt')
rijecnik = {}
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

counter = 0
for rj in rijecnik:
    if(rijecnik[rj] == 1):
        counter = counter + 1

print('Broj rijeci koje se pojavljuju samo jednom: ', counter)


#zadatak 5

with open('SMSSpamCollection.txt', 'r', encoding='utf-8') as file:
    sms_data = file.readlines()

ham_count = 0
ham_wordcount = 0
spam_count = 0
spam_wordcount = 0
counter = 0

for line in sms_data:
    label, message = line.split(None, 1)
    if(label == 'ham'):
        ham_count = ham_count + 1
        ham_wordcount += len(message.split())
    elif(label == 'spam'):
        spam_count = spam_count + 1
        spam_wordcount += len(message.split())
        if(line.endswith('!')):
            counter += 1

print('Prosjecan broj rijeci u ham porukama: ', ham_wordcount/ham_count)
print('Prosjecan broj rijeci u spam porukama: ', spam_wordcount/spam_count)
print('Broj SMS poruka koje zavrsavaju s !: ', counter)
