traces_data = []

    	tree = ET.parse(inkml_file_abs_path)
    	root = tree.getroot()
    	doc_namespace = "{http://www.w3.org/2003/InkML}"

    	'Stores traces_all with their corresponding id'
    	traces_all = [{'id': trace_tag.get('id'),
    					'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
    									for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
    								else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) \
    									for axis_coord in coord.split(' ')] \
    							for coord in (trace_tag.text).replace('\n', '').split(',')]} \
    							for trace_tag in root.findall(doc_namespace + 'trace')]
