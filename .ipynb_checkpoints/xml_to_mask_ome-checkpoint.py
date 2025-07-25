import numpy as np
import sys
import lxml.etree as ET
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation,binary_erosion
from skimage.morphology import diamond
import time

def parse_xml_annotation_mapping(xml_file):
    """
    Parse XML file and create mapping from annotation names to IDs
    Returns: dictionary {name: id}
    """
    import xml.etree.ElementTree as ET
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        annotation_mapping = {}
        
        # Find all Annotation elements
        for annotation in root.findall('.//Annotation'):
            annotation_id = annotation.get('Id')
            
            # Look for Attributes within this annotation
            attributes = annotation.find('Attributes')
            if attributes is not None:
                for attribute in attributes.findall('Attribute'):
                    name = attribute.get('Name')
                    if name and annotation_id:
                        # Clean up the name (remove leading/trailing spaces)
                        clean_name = name.strip()
                        if clean_name:  # Only add non-empty names
                            annotation_mapping[clean_name] = int(annotation_id)
        
        #print(f"Found {len(annotation_mapping)} annotations:")
        '''
        for name, id_val in sorted(annotation_mapping.items(), key=lambda x: x[1]):
            print(f"  '{name}': {id_val}")
        '''    
        return annotation_mapping
    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return {}

def get_num_classes(xml_path):
    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotation_num = 0
    for Annotation in root.findall("./Annotation"): # for all annotations
        annotation_num = annotation_num + 1

    return annotation_num + 1


"""
location (tuple) - (x, y) tuple giving the top left pixel in the level 0 reference frame
size (tuple) - (width, height) tuple giving the region size

"""

def xml_to_mask(xml_path, location, size, id_list,downsample_factor=1, verbose=0,use_six=True):
    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # calculate region bounds
    bounds = {'x_min' : location[0], 'y_min' : location[1], 'x_max' : location[0] + size[0], 'y_max' : location[1] + size[1]}

    IDs,annots = regions_in_mask(xml_path, root=root, bounds=bounds, verbose=verbose)

    if verbose != 0:
        print('\nFOUND: ' + str(len(IDs)) + ' regions')

    # fill regions and create mask
    mask = Regions_to_mask(Regions=annots, IDs=IDs,bounds=bounds,use_six=use_six, IDs_use=id_list, verbose=verbose)
    if verbose != 0:
        print('done...\n')

    return mask

def restart_line(): # for printing labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def regions_in_mask(xml_path, root, bounds, verbose=1):
    # find regions to save
    annotation_mapping = parse_xml_annotation_mapping(xml_path)
    annots=[]
    annots_IDs=[]
    for Annotation in root.findall("./Annotation"): # for all annotations
        annotationID = Annotation.attrib['Id']
        #print(annotationID)
        regs=[]
        reg_IDs=[]
        for Region in Annotation.findall("./*/Region"): # iterate on all region
            verts_x=[]
            verts_y=[]
            for Vertex in Region.findall("./*/Vertex"): # iterate on all vertex in region
                # get points

                x_point = np.int32(np.float64(Vertex.attrib['X']))
                y_point = np.int32(np.float64(Vertex.attrib['Y']))
                # test if points are in bounds
                verts_x.append(x_point)
                verts_y.append(y_point)

                #IDs.append({'regionID' : Region.attrib['Id'], 'annotationID' : annotationID})
                #break
            points = [(int(x),int(y)) for x,y in zip(verts_x,verts_y)]
            contour = np.array(points,dtype=np.int32).reshape(-1,1,2)
            area = cv2.contourArea(contour)
            if annotationID in [str(annotation_mapping["non_globally_sclerotic_glomeruli"]),str(annotation_mapping["globally_sclerotic_glomeruli"])] and area <1600: #gloms
                continue
            if annotationID == str(annotation_mapping["muscular_vessels"]) and area <1000: #muscular
                continue
            reg_IDs.append({'regionID' : Region.attrib['Id'], 'annotationID' : annotationID})
            regs.append(np.swapaxes(np.array((verts_x,verts_y)),0,1))

        annots_IDs.append(reg_IDs)
        annots.append(regs)


    return annots_IDs, annots

def Regions_to_mask(Regions, IDs,bounds,use_six, IDs_use,verbose=1):

    mask = np.zeros([ int(np.round((bounds['y_max'] - bounds['y_min']) )), int(np.round((bounds['x_max'] - bounds['x_min']) )) ], dtype=np.int8)
    #mask_temp = np.zeros([ int(np.round((bounds['y_max'] - bounds['y_min']) )), int(np.round((bounds['x_max'] - bounds['x_min']) )) ], dtype=np.int8)

    if verbose !=0:
        print('\nMAKING MASK:')

    for annot_idx,annot in enumerate(Regions):
        annot_id=IDs[annot_idx]
        for reg_idx,reg in enumerate(annot):
            reg_id=int(annot_id[reg_idx]['annotationID'])

            if reg_id not in IDs_use:
                # print(f'skpping: {reg_id}')
                continue
            x1 = min(reg[:,0])
            x2 = max(reg[:,0])
            y1 = min(reg[:,1])
            y2 = max(reg[:,1])
            reg_pass=[0,0,0,0]
            if reg_id==5:

                #for v in range(0,len(reg[1])):
                reg[:,0]=reg[:,0]-x1
                reg[:,1]=reg[:,1]-y1

                mask_temp=np.zeros((y2-y1,x2-x1))
                cv2.fillPoly(mask_temp,[reg], reg_id)

                tub_prev=mask[y1:y2,x1:x2]
                overlap=np.logical_and(tub_prev==reg_id,binary_dilation(mask_temp==reg_id,diamond(2)))
                tub_prev[mask_temp==reg_id]=reg_id
                if np.sum(overlap)>0:
                    tub_prev[overlap]=1
                    '''
                    plt.subplot(131),plt.imshow(mask_temp)
                    plt.subplot(132),plt.imshow(tub_prev)
                    plt.subplot(133),plt.imshow(overlap)
                    plt.show()
                    '''
                mask[y1:y2,x1:x2]=tub_prev

            elif reg_id==6:
                if use_six:
                    cv2.fillPoly(mask, [reg], reg_id)
                else:
                    continue
            else:
                cv2.fillPoly(mask, [reg], reg_id)

    return mask