def inkml2tag(self,  inkml_path):
		tree = ET.parse(inkml_path)
		root = tree.getroot()
		prefix = "{http://www.w3.org/2003/InkML}"
		GT_tag = [GT for GT in root.findall(prefix + 'annotation') if GT.attrib == {'type': 'truth'}]
		if GT_tag is None or len(GT_tag) == 0:
			return ""
		if GT_tag[0] is None or GT_tag[0].text is None:
			return ""
		return GT_tag[0].text
