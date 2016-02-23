import numpy as np
menNames = ["Ajinkya ","pacino","Amit","Fred","Jap","Niko","Samuel","Walt","Ari","Arrow","Arun","Bale","Bhargav","Cruise","DadPatil","DC","Dicap","Doctor","Goswami","Harvey","JE","JK","Kapish","KL","leo","loki","louis","Malkan","Manas","Vaas","mathur","michael","nicholson","Nishant","old","Puneet","RandomMan","Rishi","robert","rohit","simmons","toby","Tushar","tyrion"]
womenNames = ["akansha","arzoo","emily","isha","lorelai","mompatil","olivia","pallavi","paris","pdb","pooja","pritha","sayalee","tanvi","young","woman"]
menUniqueNames = np.sort(np.unique(menNames))
womenUniqueNames = np.sort(np.unique(womenNames))
uniqueNames = np.append(menUniqueNames, womenUniqueNames)
print(len(menUniqueNames), menUniqueNames)
print(len(womenUniqueNames), womenUniqueNames)
print(len(uniqueNames), uniqueNames)