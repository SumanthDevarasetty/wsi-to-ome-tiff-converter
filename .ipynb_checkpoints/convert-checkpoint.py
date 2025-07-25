import numpy as np
import pandas as pd
import os
import sys
import lxml.etree as ET
import tiffslide as openslide
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation,binary_erosion
from skimage.morphology import diamond
from skimage import measure
import tifffile as ti
import ome_types as ot
from glob import glob
from tqdm import tqdm
import time
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2hsv
from skimage.filters import gaussian
from xml_to_mask_ome import xml_to_mask
from skimage.transform import resize

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
        
        print(f"Found {len(annotation_mapping)} annotations:")
        for name, id_val in sorted(annotation_mapping.items(), key=lambda x: x[1]):
            print(f"  '{name}': {id_val}")
            
        return annotation_mapping
    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return {}

def apply_multiple_bounding_boxes(arr, bounding_boxes):
    """
    Apply multiple bounding boxes to create a mask that includes all tissue regions
    """
    mask = np.zeros_like(arr, dtype=bool)
    
    for bbox in bounding_boxes:
        min_row, min_col, max_row, max_col = bbox
        mask[min_row:max_row, min_col:max_col] = True
    
    arr_masked = np.where(mask, arr, 0)
    return arr_masked

def run_convert(filepath):
    """Convert OME-TIFF to segmentations format with pyramids"""
    file_ome_tif = filepath
    file_output = filepath.split('.ome')[0] + '.segmentations.ome.tiff'
    file_ome_xml = filepath.split('.tiff')[0]+ '.xml'

    cmdstring = 'tiffcomment -set '+ file_ome_xml + ' ' + file_ome_tif
    cmdstring2 = 'BF_MAX_MEM=32000M bfconvert -tilex 512 -tiley 512 -pyramid-resolutions 6 -pyramid-scale 2 -compression LZW '+filepath + ' ' + file_output
    #cmdstring4 = 'vips tiffsave'+ filepath + file_output + '--tile --pyramid --compression deflate --tile-width 512 --tile-height 512 --subifd'
    cmdstring3 = 'rm ' + filepath

    os.system(cmdstring)
    os.system(cmdstring2)
    os.system(cmdstring3)

    return file_output

def get_tissue_regions(sl, x, y, area_thresh=1000):
    """Detect tissue regions from slide thumbnail"""
    thumbIm = np.array(sl.get_thumbnail((int(x/8),int(y/8))))
    hsv = rgb2hsv(thumbIm)
    g = gaussian(hsv[:,:,1],3)
    binary = (g>0.05).astype('bool')
    binary = binary_fill_holes(binary)
    
    tissue = measure.label(binary)
    tissue_props = measure.regionprops(tissue)
    
    tissue_areas = [tissue_props[x].area for x in range(len(tissue_props))]
    tissue_idx = list(range(len(tissue_props)))
    tissue_idx = [tissue_idx[x] for x in range(len(tissue_props)) if tissue_areas[x] > area_thresh]
    
    if len(tissue_idx)==0:
        return None, None, None
    
    tissue_centroids = [tissue_props[x].centroid for x in tissue_idx]
    
    return tissue_props, tissue_idx, tissue_centroids, binary

def get_all_tissue_bounding_boxes(tissue_props, tissue_idx, tissue_centroids):
    """
    Get bounding boxes for ALL tissue sections without QC filtering
    Returns a list of bounding boxes for all valid tissue regions
    """
    all_bboxes = []
    
    print(f"Found {len(tissue_idx)} tissue regions:")
    
    for i, prop_idx in enumerate(tissue_idx):
        bbox = tissue_props[prop_idx].bbox
        area = tissue_props[prop_idx].area
        centroid = tissue_props[prop_idx].centroid
        
        all_bboxes.append(bbox)
        print(f"  Region {i+1}: bbox={bbox}, area={area:,}, centroid={centroid}")
    
    return all_bboxes

def load_and_process_annotations_all_regions(nm, x, y, all_bboxes_scaled, slide_id, channel_names):
    #1- cortex, 2- medulla (x), 3/4 - Gloms, 5 - tubules, 6 - 
    """Load all annotation types and process them"""
    
    # Initialize output arrays
    full = np.zeros((len(channel_names), y, x), dtype=np.uint16)
    obj_num = []
    channel_num = []
    ont_nm = []
    channel_nm = []
    
    ontology_names = ['0001225','0000073','0000074','0009773','0001637','0006851','0005215']

    annotation_mapping = parse_xml_annotation_mapping(nm)

    # LOAD CORTEX
    #mask = xml_to_mask(nm,(0,0),(x,y),[1])
    mask = xml_to_mask(nm,(0,0),(x,y),[annotation_mapping["cortical_interstitium"]])
    mask = mask.astype(np.uint8)
    cortex = apply_multiple_bounding_boxes(mask, all_bboxes_scaled)
    obj_num.append(1)
    channel_num.append(0)
    channel_nm.append(channel_names[0])
    ont_nm.append(ontology_names[0])
    full[0,:,:] = cortex
    
    # MultiC Annotations 
    #mask = xml_to_mask(nm,(0,0),(x,y),[3,4,5,6])
    mask = xml_to_mask(nm,(0,0),(x,y), [annotation_mapping["non_globally_sclerotic_glomeruli"],annotation_mapping["globally_sclerotic_glomeruli"],annotation_mapping["tubules"],annotation_mapping["muscular_vessels"]])
    mask = mask.astype(np.uint8)
    multic = apply_multiple_bounding_boxes(mask, all_bboxes_scaled)
    print("Shape of multic:", multic.shape)
    unique_values = np.unique(multic)
    print("Unique annotation values in multic:", unique_values)
    
    # Process glomeruli
    #glom = (multic==3).astype(np.uint8)
    glom = (multic==annotation_mapping["non_globally_sclerotic_glomeruli"]).astype(np.uint8)
    glom = measure.label(glom)
    obj_num.append(np.max(glom))
    channel_num.append(1)
    channel_nm.append(channel_names[1])
    ont_nm.append(ontology_names[1])
    full[1,:,:] = glom
    del glom
    
    #sglom = (multic==4).astype(np.uint8)
    sglom = (multic==annotation_mapping["globally_sclerotic_glomeruli"]).astype(np.uint8)
    sglom = measure.label(sglom)
    #sglom2 = sglom + n_gloms
    #sglom2[sglom==0]=0
    obj_num.append(np.max(sglom))
    channel_num.append(2)
    channel_nm.append(channel_names[2])
    ont_nm.append(ontology_names[2])
    full[2,:,:] = sglom
    del sglom
    
    #sglom=sglom2
    #del sglom2
    
    # Process tubules
    #tub = (multic==5).astype(np.uint8)
    tub = (multic==annotation_mapping["tubules"]).astype(np.uint8)
    tub = measure.label(tub)
    obj_num.append(np.max(tub))
    channel_num.append(3)
    channel_nm.append(channel_names[3])
    ont_nm.append(ontology_names[3])
    full[3,:,:] = tub
    del tub
    
    # Process arteries
    #art = (multic==6).astype(np.uint8)
    art = (multic==annotation_mapping["muscular_vessels"]).astype(np.uint8)
    art = measure.label(art)
    obj_num.append(np.max(art))
    channel_num.append(4)
    channel_nm.append(channel_names[4])
    ont_nm.append(ontology_names[4])
    full[4,:,:] = art
    del art,multic
    
    # PTC
    #mask = xml_to_mask(nm,(0,0),(x,y),[7])
    #mask = xml_to_mask(nm,(0,0),(x,y),[annotation_mapping["peritubular_capillaries"]])
    ptc_id = annotation_mapping.get("peritubular_capillaries", annotation_mapping.get("ptc", None))
    mask = xml_to_mask(nm,(0,0),(x,y),[ptc_id])
    mask = mask.astype(np.uint8)
    #ptc = apply_bounding_box(mask,bbox_oi)
    ptc = apply_multiple_bounding_boxes(mask, all_bboxes_scaled)
    #ptc = ptc*cortex
    ptc = measure.label(ptc)
    obj_num.append(np.max(ptc))
    channel_num.append(5)
    channel_nm.append(channel_names[5])
    ont_nm.append(ontology_names[5])
    full[5,:,:] = ptc
    del ptc,cortex
    
    # IFTA
    #mask = xml_to_mask(nm,(0,0),(x,y),[8])
    #mask = xml_to_mask(nm,(0,0),(x,y),[annotation_mapping["IFTA"]])
    # IFTA - Handle missing annotation
    ifta_key = annotation_mapping.get("IFTA", annotation_mapping.get("ifta", None))
    if ifta_key:
        mask = xml_to_mask(nm,(0,0),(x,y),[ifta_key])
    else:
        print("Warning: No IFTA annotation found in XML")
        mask = np.zeros((y, x), dtype=np.uint8)
    
    mask = mask.astype(np.uint8)
    #ifta = apply_bounding_box(mask,bbox_oi)
    ifta = apply_multiple_bounding_boxes(mask, all_bboxes_scaled)
    '''
    if slide_id == 'S-2010-012854':
        print('Loading 2nd IFTA in: {}'.format(slide_id))
        ifta2 = xml_to_mask(nm.split('.xml')[0]+'_IFTA2.xml',(0,0),(x,y),[1])
        ifta2 = ifta2.astype(np.uint8)
        #ifta2 = apply_bounding_box(ifta2, bbox_oi)
        ifta2 = apply_multiple_bounding_boxes(ifta2, all_bboxes_scaled)
        ifta = ifta+ifta2
    '''
    obj_num.append(1)
    channel_num.append(6)
    channel_nm.append(channel_names[6])
    ont_nm.append(ontology_names[6])
    
    full[6,:,:] = ifta
    del ifta
    
    return full, obj_num, channel_num, ont_nm, channel_nm

def create_csv_data(channel_num, channel_nm, slide, ont_nm, obj_num, protocol_doi):
    """Create CSV data for metadata"""
    csv_lines = []
    
    for channels in range(len(channel_num)):
        current_line = []
        current_line.append(channel_num[channels])
        current_line.append(channel_nm[channels])
        current_line.append(slide.split('/')[-1].split('.')[0] + '.segmentations.ome.tiff')
        current_line.append('Uberon')
        current_line.append(ont_nm[channels])
        current_line.append(protocol_doi)
        current_line.append('Yes')
        current_line.append(obj_num[channels])
        csv_lines.append(current_line)
    
    return csv_lines

def write_ome_tiff(slide_name, full, channel_names, save_dir, slide):
    """Write OME-TIFF file with metadata"""
    
    # Write TIFF
    tiff_writer = ti.TiffWriter(slide_name,ome=True,bigtiff=True)
    tiff_writer.write(full,metadata={'axes':'CYX'})
    tiff_writer.close()

    # Update metadata
    tiff_file = ot.from_tiff(slide_name)
    full_list = channel_names
    print("Order in ome tiff: ")
    '''
    for i,c in enumerate(tiff_file.images[0].pixels.channels):
        c.name = full_list[i]
        print(full_list[i])
    '''
    print("Setting channel names in OME metadata:")
    for i,c in enumerate(tiff_file.images[0].pixels.channels):
        c.name = full_list[i]
        print(f"  Channel {i}: {full_list[i]}")

    # Write XML metadata
    xml_name = save_dir + slide.split('/')[-1].split('.')[0] + '.ome.xml'
    xml_data = ot.to_xml(tiff_file)
    xml_data = xml_data.replace('<Pixels','<Pixels PhysicalSizeXUnit="\u03BCm" PhysicalSizeYUnit="\u03BCm"')
    with open(xml_name,'wt+') as fh:
        fh.write(xml_data)

def process_single_slide_no_qc(slide, save_dir, channel_names, area_thresh, protocol_doi):
    """Process a single slide through the entire pipeline"""
    
    #pad = 100
    slide_name = save_dir + slide.split('/')[-1].split('.')[0] + '.ome.tiff'
    seg_name = slide_name.split('.ome')[0] + '.segmentations.ome.tiff'

    if os.path.exists(seg_name):
        print('Skipping: {}'.format(seg_name))
        return None

    sl = openslide.OpenSlide(slide)
    x,y = sl.dimensions
    x=int(x)
    y=int(y)

    nm = slide.rsplit(".", 1)[0]+'.xml'
    if not os.path.exists(nm):
        return None
        
    slide_id = slide.split('/')[-1].split('_')[0]
    
    if os.path.exists(save_dir + slide.split('/')[-1].split('.')[0] + '_cortex.png'):
        print('Already done with: {}'.format(slide_id))
        return None
    
    print("slide_id", slide_id)
    
    # Get tissue regions
    result = get_tissue_regions(sl, x, y, area_thresh)
    if result is None:
        print('Failed on: {}'.format(slide_id))
        return None
    
    tissue_props, tissue_idx, tissue_centroids, binary = result
    
    # Now gets ALL tissue bounding boxes:
    all_bboxes = get_all_tissue_bounding_boxes(tissue_props, tissue_idx, tissue_centroids)
    
    # Scale all bounding boxes to full resolution
    all_bboxes_scaled = [(8*bbox[0], 8*bbox[1], 8*bbox[2], 8*bbox[3]) for bbox in all_bboxes]
    
    print("Scaled bounding boxes:")
    for i, bbox in enumerate(all_bboxes_scaled):
        print(f"  Region {i+1}: {bbox}")
    
    # Adjust padding
    #pad = adjust_padding(bbox_oi, binary, pad)
    
    # Scale bounding box to full resolution
    #bbox_oi = (8*bbox_oi[0],8*bbox_oi[1],8*bbox_oi[2],8*bbox_oi[3])
    
    # Load and process annotations
    #full, obj_num, channel_num, ont_nm, channel_nm = load_and_process_annotations(nm, x, y, bbox_oi, slide_id, channel_names)
    
    # Load and process annotations for all regions
    full, obj_num, channel_num, ont_nm, channel_nm = load_and_process_annotations_all_regions(nm, x, y, all_bboxes_scaled, slide_id, channel_names)
    
    # Create CSV data
    csv_lines = create_csv_data(channel_num, channel_nm, slide, ont_nm, obj_num, protocol_doi)
    
    # Write OME-TIFF
    write_ome_tiff(slide_name, full, channel_names, save_dir, slide)
    
    # Convert to final format
    output_name = run_convert(slide_name)
    print(output_name)
    
    return csv_lines


def main():
    """Main processing function"""
    
    # Configuration
    wsi_ext = ['.svs','.scn']
    channel_names = ['cortex','non_globally_sclerotic_glomeruli','globally_sclerotic_glomeruli','tubules','muscular_vessels','peritubular_capillaries','IFTA']
    ontology_names = ['0001225','0000073','0000074','0009773','0001637','0006851','0005215']

    slide_dir = '/blue/pinaki.sarder/sdevarasetty/KPMP/kpmp_test/'
    save_dir = '/blue/pinaki.sarder/sdevarasetty/KPMP/kpmp_convert_test/sumanth/'
    
    area_thresh = 1000
    csv_file = 'segmentation-masks.csv'
    csv_columns = ['Channel number','Mask name','Source file','Ontology abbreviation','Ontology ID','Protocol','Is entire image masked','Num objects']
    protocol_doi = 'dx.doi.org/10.17504/protocols.io.dm6gp35p8vzp/v1'

    # Get slides
    slides = []
    for ext in wsi_ext:
        slides.extend(glob(slide_dir + '*' + ext))
    si=-1
    soi = []
    csv_lines = []

    print(f"Processing {len(slides)} slides without QC data...")
    print("Will process ALL tissue regions found in each slide")

    # Process each slide
    for slide in tqdm(slides):
        si+=1
        
        if slide in soi:
            continue
        
        # Process single slide
        #slide_csv_lines = process_single_slide(slide, save_dir, qc_data, double_slides, channel_names, area_thresh, protocol_doi)
        slide_csv_lines = process_single_slide_no_qc(slide, save_dir, channel_names, area_thresh, protocol_doi)

        if slide_csv_lines is not None:
            csv_lines.extend(slide_csv_lines)
    # Save final CSV
    csv_df = pd.DataFrame(csv_lines,columns=csv_columns)
    csv_df.to_csv(save_dir + csv_file,index=False)
    
if __name__ == "__main__":
    main()