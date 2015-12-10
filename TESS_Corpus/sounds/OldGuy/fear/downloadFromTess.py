import urllib

def main():
	young = "YAF"
	old = "OAF"
	oldEmotions = {"ps":24491, "angry":24499, "disgust":24494, "fear":24492, "happy":24501, "neutral":24488, "sad":24497}
	youngEmotions = {"ps":24495, "angry":24490, "disgust":24498, "fear":24489, "happy":24493, "neutral":24496, "sad":24500}
	words = ["youth", "young", "yes", "yearn", "witch", "wire", "wife", "white", "whip", "which", "when", "wheat", "week", "wash", "walk", "wag", "vote", "void", "voice", "vine", "turn", "tough", "tool", "ton", "tire", "tip", "time", "thumb", "thought", "third", "thin", "tell", "team", "tape", "talk", "take", "sure", "such", "sub", "south", "sour", "soup", "soap", "size", "shout", "should", "shirt", "sheep", "shawl", "shall", "shack", "sell", "seize", "search", "sail", "said", "rush", "rough", "rot", "rose", "room", "road", "ripe", "ring", "red", "read", "reach", "rat", "raise", "rain", "raid", "rag", "puff", "pool", "pole", "pike", "pick", "phone", "perch", "peg", "pearl", "pass", "pain", "page", "pad", "numb", "note", "nice", "neat", "near", "name", "nag", "mouse", "mop", "moon", "mood", "mode", "mob", "mill", "met", "mess", "merge", "match", "make", "luck", "love", "lot", "lose", "lore", "long", "loaf", "live", "limb", "life", "lid", "lease", "learn", "lean", "laud", "late", "knock", "kite", "king", "kill", "kick", "keg", "keep", "keen", "juice", "jug", "judge", "join", "jar", "jail", "hush", "hurl", "home", "hole", "hit", "hire", "haze", "have", "hate", "hash", "hall", "half", "gun", "goose", "good", "goal", "gin", "get", "germ", "gaze", "gas", "gap", "food", "five", "fit", "fat", "far", "fall", "fail", "door", "doll", "dog", "dodge", "ditch", "dip", "dime", "deep", "death", "dead", "date", "dab", "cool", "choice", "chief", "cheek", "check", "chat", "chalk", "chair", "chain", "cause", "came", "calm", "cab", "burn", "bought", "book", "bone", "boat", "bite", "beg", "bean", "bath", "base", "bar", "back"]
	i = 1
	keys = oldEmotions.keys()
	i = 1
	while i < 200:
		url = "https://tspace.library.utoronto.ca/bitstream/1807/" + str(24492) + "/" + str((i + 1)) + "/OAF_" + str(words[i]) + "_fear.wav"
		fileName = "OAF_" + str(words[i]) + "_fear.wav"
		print("Downloading: " + fileName)
		urllib.urlretrieve(url, fileName)
		i = i + 1

	# keys = youngEmotions.keys()
	# for tempEmote in keys:
	# 	i = 1
	# 	while i < 200:
	# 		url = "https://tspace.library.utoronto.ca/bitstream/1807/" + str(youngEmotions[tempEmote]) + "/" + str((i + 1)) + "/YAF_" + str(words[i]) + "_" + str(tempEmote)  + ".wav"
	# 		fileName = "YAF_" + str(words[i]) + "_" + str(tempEmote)  + ".wav"
	# 		print("Downloading: " + fileName + " from: " + url)
	# 		urllib.urlretrieve(url, fileName)
	# 		i = i + 1

if __name__ == "__main__":
    main()