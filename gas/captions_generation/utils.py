def make_and_description(names):
	names = [normalized_name(name) for name in names]
	if len(names) == 0:
		return ""
	elif len(names) == 1:
		return names[0]
	elif len(names) == 2:
		return ' and '.join(names)
	else:
		names = names[:-1] + [f'and {names[-1]}']
		return ', '.join(names)

def normalized_name(name):
	return name.replace('_', ' ')