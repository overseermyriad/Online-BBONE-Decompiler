import os
import sys
import json
import shutil
import struct
import uuid
import zlib
import hashlib
import re
import argparse
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import Dict, List, Any, BinaryIO

try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_FILTER = Image.LANCZOS

TYPE_HAS_XY = 1
TYPE_HAS_M_A = 2
TYPE_HAS_M_B = 4
TYPE_HAS_M_C = 8
TYPE_HAS_M_D = 16
TYPE_HAS_ALPHA = 32
TYPE_HAS_COLORTRANSFORM = 64
TYPE_HAS_BLENDMODE = 128
TYPE_HAS_CHILDREN = 256
TYPE_HAS_BATCHES = 512

def read_short(stream: BinaryIO) -> int:
    return struct.unpack('>h', stream.read(2))[0]

def read_ushort(stream: BinaryIO) -> int:
    return struct.unpack('>H', stream.read(2))[0]

def read_int(stream: BinaryIO) -> int:
    return struct.unpack('>i', stream.read(4))[0]

def read_float(stream: BinaryIO) -> float:
    return struct.unpack('>f', stream.read(4))[0]

def read_utf(stream: BinaryIO) -> str:
    length = read_ushort(stream)
    return stream.read(length).decode('utf-8')

def gen_uuid():
    return f"{uuid.uuid4().hex[:8]}-{uuid.uuid4().hex[8:16]}"

def sanitize_name(name: str) -> str:
    safe = re.sub(r'[\\/*?:"<>|\[\]=,\(\)\s]', "_", name)
    if len(safe) > 50:
        hash_str = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
        safe = safe[:40] + "_" + hash_str
    return safe

def unzip_bbone(file_path):
    with open(file_path, "rb") as f:
        data = f.read() 

    object_name = str(Path(file_path).name).split(".")[0]

    if data[0] == 0x78:
        uncompressed = zlib.decompress(data)
        version = 0 
    else:
        file_type = struct.unpack(">H", data[0:2])[0]
        if file_type not in (0x5678, 0x5679):
            raise Exception("Not a valid BBONE file")

        header_len = struct.unpack(">H", data[2:4])[0]
        version = struct.unpack(">H", data[4:6])[0] 

        compressed_data = data[header_len:]
        uncompressed = zlib.decompress(compressed_data)

    plugin_map = []
    ptr = 0
    while True:
        plugin_id = uncompressed[ptr]
        ptr += 1
        if plugin_id == 0:
            break
        offset = struct.unpack(">I", uncompressed[ptr:ptr+4])[0]
        ptr += 4
        length = struct.unpack(">I", uncompressed[ptr:ptr+4])[0]
        ptr += 4
        plugin_map.append({"id": plugin_id, "offset": offset, "length": length})

    plugin_data_blob = uncompressed[ptr:]
    plugin_contents = {}
    for plugin in plugin_map:
        raw = plugin_data_blob[plugin["offset"]:plugin["offset"] + plugin["length"]]
        plugin_contents[plugin["id"]] = raw

    return object_name, version, plugin_contents

def process_plugin_1(data: bytes, save_dir: str):
    if not data: return []
    stream = BytesIO(data)
    bitmap_flag = struct.unpack("B", stream.read(1))[0]
    
    plist = []
    
    if bitmap_flag == 0xFF:
        bitmap_count = read_ushort(stream)
        bitmaps = []
        for _ in range(bitmap_count):
            width = read_ushort(stream)
            height = read_ushort(stream)
            marker = read_ushort(stream)
            if marker == 65495: 
                size = read_int(stream)
                jpeg_data = stream.read(size)
                image = Image.open(BytesIO(jpeg_data)).convert("RGBA")
            else:
                stream.seek(-2, 1)
                pixel_data = stream.read(width * height * 4)
                image = Image.frombytes("RGBA", (width, height), pixel_data)
                arr = np.array(image)
                if arr.shape[2] == 4:
                    arr = arr[..., [1, 2, 3, 0]]
                image = Image.fromarray(arr, "RGBA")
            bitmaps.append(image)
        
        atlas = bitmaps[0] if bitmaps else None
        
        frame_count = read_ushort(stream)
        for _ in range(frame_count):
            name_len = read_ushort(stream)
            raw_name = stream.read(name_len).decode("utf-8")
            name = sanitize_name(raw_name)

            total_frames = read_int(stream)
            for _ in range(total_frames):
                frame_type = struct.unpack("B", stream.read(1))[0]
                if frame_type == 0xFF:
                    bitmapDataId = read_ushort(stream)
                    offsetX = read_short(stream)
                    offsetY = read_short(stream)
                    width = read_ushort(stream)
                    height = read_ushort(stream)
                    xOrg = read_float(stream)
                    yOrg = read_float(stream)
                    scaleX = read_float(stream)
                    scaleY = read_float(stream)
                    rotation = read_float(stream)

                    plist.append({
                        "name": name,
                        "bitmap_id": bitmapDataId,
                        "rect_x": offsetX,
                        "rect_y": offsetY,
                        "rect_w": width,
                        "rect_h": height,
                        "origin_x": xOrg,
                        "origin_y": yOrg,
                        "scale_x": scaleX,
                        "scale_y": scaleY,
                        "rotation": rotation
                    })
                else:
                    break
                    
        if atlas:
            for entry in plist:
                entry_name = entry['name']
                x = entry['rect_x']
                y = entry['rect_y']
                w = entry['rect_w']
                h = entry['rect_h']
                
                cropped = atlas.crop((x, y, x + w, y + h))
                save_path = os.path.join(save_dir, f"{entry_name}.png")
                cropped.save(save_path)
            
    else:
        stream.seek(-1, 1)
        blit_count = read_int(stream)
        
        for _ in range(blit_count):
            name_len = read_ushort(stream)
            raw_name = stream.read(name_len).decode("utf-8")
            name = sanitize_name(raw_name)
            
            total_frames = read_int(stream)
            for f_idx in range(total_frames):
                frame_type = struct.unpack("B", stream.read(1))[0]
                if frame_type == 0xFF:
                    bitmapDataId = read_short(stream)
                    offsetX = read_short(stream)
                    offsetY = read_short(stream)
                    w = read_short(stream)
                    h = read_short(stream)
                    xOrg = read_float(stream)
                    yOrg = read_float(stream)
                    scaleX = read_float(stream)
                    scaleY = read_float(stream)
                    rotation = read_float(stream)
                else:
                    stream.seek(-1, 1)
                    w = read_int(stream)
                    h = read_int(stream)
                    pixel_data = stream.read(w * h * 4)
                    
                    image = Image.frombytes("RGBA", (w, h), pixel_data)
                    arr = np.array(image)
                    if arr.shape[2] == 4:
                        arr = arr[..., [1, 2, 3, 0]]
                    image = Image.fromarray(arr, "RGBA")
                    
                    x = struct.unpack(">d", stream.read(8))[0]
                    y = struct.unpack(">d", stream.read(8))[0]
                    scaleX = struct.unpack(">d", stream.read(8))[0]
                    scaleY = struct.unpack(">d", stream.read(8))[0]
                    rotation = struct.unpack(">d", stream.read(8))[0]
                    
                    frame_name = f"{name}_{f_idx}" if total_frames > 1 else name
                    image.save(os.path.join(save_dir, f"{frame_name}.png"))
                    
                    plist.append({
                        "name": frame_name,
                        "bitmap_id": 0,
                        "rect_x": 0,
                        "rect_y": 0,
                        "rect_w": w,
                        "rect_h": h,
                        "origin_x": float(x),
                        "origin_y": float(y),
                        "scale_x": float(scaleX),
                        "scale_y": float(scaleY),
                        "rotation": float(rotation)
                    })
                    
    return plist

def parse_frame_labels(data: bytes):
    if not data: return {}
    pos = 0
    labels = {}
    count = struct.unpack_from(">I", data, pos)[0]
    pos += 4

    for _ in range(count):
        name_len = struct.unpack_from(">H", data, pos)[0]
        pos += 2
        name_bytes = data[pos:pos + name_len]
        try:
            name = name_bytes.decode('ascii')
        except UnicodeDecodeError:
            name = name_bytes.decode('latin1')
            
        name = sanitize_name(name)
        pos += name_len

        frame = struct.unpack_from(">I", data, pos)[0]
        pos += 4
        labels[name] = frame

    return labels

def read_color_transform(stream: BinaryIO) -> Dict[str, float]:
    return {
        "alphaMultiplier": read_float(stream),
        "alphaOffset": read_float(stream),
        "blueMultiplier": read_float(stream),
        "blueOffset": read_float(stream),
        "greenMultiplier": read_float(stream),
        "greenOffset": read_float(stream),
        "redMultiplier": read_float(stream),
        "redOffset": read_float(stream),
    }

def parse_child_node(stream: BinaryIO, shared_pool: Dict) -> Dict[str, Any]:
    flags = read_short(stream)
    class_name = read_utf(stream)
    class_name = sanitize_name(class_name)

    node = {
        "name": class_name,
        "matrix": {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0, "tx": 0.0, "ty": 0.0},
        "color": {},
        "children": []
    }

    if flags & TYPE_HAS_XY:
        node["matrix"]["tx"] = read_float(stream)
        node["matrix"]["ty"] = read_float(stream)
    
    if flags & TYPE_HAS_M_A: node["matrix"]["a"] = read_float(stream)
    if flags & TYPE_HAS_M_B: node["matrix"]["b"] = read_float(stream)
    if flags & TYPE_HAS_M_C: node["matrix"]["c"] = read_float(stream)
    if flags & TYPE_HAS_M_D: node["matrix"]["d"] = read_float(stream)

    if flags & TYPE_HAS_ALPHA:
        node["color"]["alphaMultiplier"] = read_float(stream)
        
    if flags & TYPE_HAS_COLORTRANSFORM:
        node["color"] = read_color_transform(stream)

    if flags & TYPE_HAS_BLENDMODE:
        node["blendMode"] = read_utf(stream)

    if flags & TYPE_HAS_CHILDREN:
        children_count = read_short(stream)
        for _ in range(children_count):
            child = parse_child_node(stream, shared_pool)
            node["children"].append(child)

    if flags & TYPE_HAS_BATCHES:
        node["references_shared_animation"] = class_name
        
    return node

def parse_single_frame_batch(stream: BinaryIO, shared_pool: Dict) -> Dict[str, List]:
    children_count = read_int(stream)
    frame = {"children": []}
    for _ in range(children_count):
        frame["children"].append(parse_child_node(stream, shared_pool))
    return frame

def decode_animation_chunk(data: bytes, version: int) -> Dict[str, Any]:
    if not data: return {"shared_animations": {}, "frames": []}
    stream = BytesIO(data)
    shared_animations_pool = {}
    
    if version >= 4:
        try:
            shared_block_count = read_short(stream)
            for _ in range(shared_block_count):
                block_name = read_utf(stream)
                block_name = sanitize_name(block_name)
                frames_in_block = read_short(stream)
                
                animation_frames = []
                for _ in range(frames_in_block):
                    frame_data = parse_single_frame_batch(stream, shared_animations_pool)
                    animation_frames.append(frame_data)
                shared_animations_pool[block_name] = animation_frames
        except struct.error:
            pass

    main_timeline_frames = []
    try:
        total_frames = read_int(stream)
        for i in range(total_frames):
            try:
                frame_data = parse_single_frame_batch(stream, shared_animations_pool)
                main_timeline_frames.append(frame_data)
            except struct.error:
                break
    except struct.error:
        pass
            
    return {
        "shared_animations": shared_animations_pool,
        "frames": main_timeline_frames
    }

def color_to_xml(color_dict):
    if not color_dict:
        return ""
        
    attrs = []
    rm = color_dict.get("redMultiplier", 1.0)
    if abs(rm - 1.0) > 0.001: attrs.append(f'redMultiplier="{rm}"')
    
    gm = color_dict.get("greenMultiplier", 1.0)
    if abs(gm - 1.0) > 0.001: attrs.append(f'greenMultiplier="{gm}"')
    
    bm = color_dict.get("blueMultiplier", 1.0)
    if abs(bm - 1.0) > 0.001: attrs.append(f'blueMultiplier="{bm}"')
    
    am = color_dict.get("alphaMultiplier", 1.0)
    if abs(am - 1.0) > 0.001: attrs.append(f'alphaMultiplier="{am}"')
    
    ro = color_dict.get("redOffset", 0.0)
    if abs(ro) > 0.001: attrs.append(f'redOffset="{round(ro)}"')
    
    go = color_dict.get("greenOffset", 0.0)
    if abs(go) > 0.001: attrs.append(f'greenOffset="{round(go)}"')
    
    bo = color_dict.get("blueOffset", 0.0)
    if abs(bo) > 0.001: attrs.append(f'blueOffset="{round(bo)}"')
    
    ao = color_dict.get("alphaOffset", 0.0)
    if abs(ao) > 0.001: attrs.append(f'alphaOffset="{round(ao)}"')

    if attrs:
        return f'<color><Color {" ".join(attrs)}/></color>'
    return ""

def matrix_to_xml(mat_dict, offset_x=0.0, offset_y=0.0):
    if not mat_dict:
        return f'<Matrix tx="{offset_x}" ty="{offset_y}"/>' if offset_x or offset_y else '<Matrix tx="0" ty="0"/>'
    
    a, b = mat_dict.get("a", 1.0), mat_dict.get("b", 0.0)
    c, d = mat_dict.get("c", 0.0), mat_dict.get("d", 1.0)
    tx = mat_dict.get("tx", 0.0) + offset_x
    ty = mat_dict.get("ty", 0.0) + offset_y
    
    return f'<Matrix a="{a}" b="{b}" c="{c}" d="{d}" tx="{tx}" ty="{ty}"/>'

def flatten_frame_children(children_list, parent_mat=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0), parent_color=None):
    if parent_color is None:
        parent_color = {}
        
    flat_list = []
    for child in children_list:
        c_mat = child.get("matrix", {})
        ca, cb = c_mat.get("a", 1.0), c_mat.get("b", 0.0)
        cc, cd = c_mat.get("c", 0.0), c_mat.get("d", 1.0)
        ctx, cty = c_mat.get("tx", 0.0), c_mat.get("ty", 0.0)
        
        pa, pb, pc, pd, ptx, pty = parent_mat
        na = pa * ca + pc * cb
        nb = pb * ca + pd * cb
        nc = pa * cc + pc * cd
        nd = pb * cc + pd * cd
        ntx = pa * ctx + pc * cty + ptx
        nty = pb * ctx + pd * cty + pty
        new_mat_tuple = (na, nb, nc, nd, ntx, nty)
        
        c_color = child.get("color", {})
        
        p_rm = parent_color.get("redMultiplier", 1.0)
        p_gm = parent_color.get("greenMultiplier", 1.0)
        p_bm = parent_color.get("blueMultiplier", 1.0)
        p_am = parent_color.get("alphaMultiplier", 1.0)
        p_ro = parent_color.get("redOffset", 0.0)
        p_go = parent_color.get("greenOffset", 0.0)
        p_bo = parent_color.get("blueOffset", 0.0)
        p_ao = parent_color.get("alphaOffset", 0.0)

        c_rm = c_color.get("redMultiplier", 1.0)
        c_gm = c_color.get("greenMultiplier", 1.0)
        c_bm = c_color.get("blueMultiplier", 1.0)
        c_am = c_color.get("alphaMultiplier", 1.0)
        c_ro = c_color.get("redOffset", 0.0)
        c_go = c_color.get("greenOffset", 0.0)
        c_bo = c_color.get("blueOffset", 0.0)
        c_ao = c_color.get("alphaOffset", 0.0)

        new_color = {
            "redMultiplier": c_rm * p_rm,
            "greenMultiplier": c_gm * p_gm,
            "blueMultiplier": c_bm * p_bm,
            "alphaMultiplier": c_am * p_am,
            "redOffset": c_ro * p_rm + p_ro,
            "greenOffset": c_go * p_gm + p_go,
            "blueOffset": c_bo * p_bm + p_bo,
            "alphaOffset": c_ao * p_am + p_ao
        }
        
        if child.get("children"):
            flat_list.extend(flatten_frame_children(child["children"], new_mat_tuple, new_color))
        else:
            leaf = {
                "name": child.get("name"),
                "matrix": {"a": na, "b": nb, "c": nc, "d": nd, "tx": ntx, "ty": nty},
                "color": new_color
            }
            if child.get("references_shared_animation"):
                leaf["references_shared_animation"] = child.get("references_shared_animation")
            flat_list.append(leaf)
    return flat_list

def compute_bounds(elements, shared_animations, plist_dict, alias_map, parent_mat=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0), depth=0):
    if depth > 20: return None
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    
    for child in elements:
        mat = child.get("matrix", {})
        a, b = mat.get("a", 1.0), mat.get("b", 0.0)
        c, d = mat.get("c", 0.0), mat.get("d", 1.0)
        tx, ty = mat.get("tx", 0.0), mat.get("ty", 0.0)
        
        pa, pb, pc, pd, ptx, pty = parent_mat
        ca, cb = pa*a + pc*b, pb*a + pd*b
        cc, cd = pa*c + pc*d, pb*c + pd*d
        ctx, cty = pa*tx + pc*ty + ptx, pb*tx + pd*ty + pty
        new_mat = (ca, cb, cc, cd, ctx, cty)
        
        if child.get("references_shared_animation"):
            shared_ref = child["references_shared_animation"]
            shared_frames = shared_animations.get(shared_ref, [])
            if shared_frames:
                first_frame_els = shared_frames[0].get("children", [])
                bnd = compute_bounds(first_frame_els, shared_animations, plist_dict, alias_map, new_mat, depth+1)
                if bnd:
                    min_x, min_y = min(min_x, bnd[0]), min(min_y, bnd[1])
                    max_x, max_y = max(max_x, bnd[2]), max(max_y, bnd[3])
        elif child.get("children"): 
            bnd = compute_bounds(child["children"], shared_animations, plist_dict, alias_map, new_mat, depth+1)
            if bnd:
                min_x, min_y = min(min_x, bnd[0]), min(min_y, bnd[1])
                max_x, max_y = max(max_x, bnd[2]), max(max_y, bnd[3])
        else:
            name = child.get("name")
            mapped_name = alias_map.get(name, name)
            if mapped_name in plist_dict:
                item = plist_dict[mapped_name]
                w, h = item.get("rect_w", 0), item.get("rect_h", 0)
                sx, sy = item.get("scale_x", 1.0), item.get("scale_y", 1.0)
                ox, oy = item.get("origin_x", 0.0), item.get("origin_y", 0.0)
                
                corners = [(0, 0), (w, 0), (w, h), (0, h)]
                for cx, cy in corners:
                    lx, ly = sx * cx + ox, sy * cy + oy
                    gx, gy = ca * lx + cc * ly + ctx, cb * lx + cd * ly + cty
                    min_x, min_y = min(min_x, gx), min(min_y, gy)
                    max_x, max_y = max(max_x, gx), max(max_y, gy)
    
    if min_x == float('inf'): return None
    return (min_x, min_y, max_x, max_y)

def build_xfl(json_data, input_png_dir, output_xfl_dir, args):
    plist = json_data.get("plist", [])
    alias_map = json_data.get("alias_map", {})
    anim_data = json_data.get("animation", {})
    frames = anim_data.get("frames", json_data.get("frames", []))
    shared_animations = anim_data.get("shared_animations", {})
    
    labels_dict = {}
    if "labels" in json_data:
        if isinstance(json_data["labels"], dict):
            labels_dict = json_data["labels"]
        elif isinstance(json_data["labels"], list):
            for lbl in json_data["labels"]:
                labels_dict[lbl["name"]] = lbl.get("frame", 0)
    elif "animationLabels" in json_data:
        labels_dict = json_data["animationLabels"]
        
    if not labels_dict:
        labels_dict = {"main_animation": 1}
        
    sorted_labels = sorted(labels_dict.items(), key=lambda x: x[1])
    animations = []
    total_frames = len(frames)
    
    for i in range(len(sorted_labels)):
        name = sorted_labels[i][0]
        start = max(0, sorted_labels[i][1] - 1) 
        end = sorted_labels[i+1][1] - 1 if i + 1 < len(sorted_labels) else total_frames
        dur = end - start
        if dur > 0:
            animations.append({'name': name, 'start': start, 'duration': dur})

    if not animations and total_frames > 0:
        animations.append({'name': "animation", 'start': 0, 'duration': total_frames})

    print("-> Calculating optimal stage centering...")
    plist_dict = {item["name"]: item for item in plist}
    shift_x, shift_y = 195.0, 195.0
    
    if animations and len(frames) > animations[0]['start']:
        probe_elements = frames[animations[0]['start']].get("children", [])
        bounds = compute_bounds(probe_elements, shared_animations, plist_dict, alias_map)
        if bounds:
            shift_x = round(195.0 - (bounds[0] + bounds[2]) * 0.5, 4)
            shift_y = round(195.0 - (bounds[1] + bounds[3]) * 0.5, 4)

    if os.path.exists(output_xfl_dir):
        shutil.rmtree(output_xfl_dir)
            
    library_dir = os.path.join(output_xfl_dir, "LIBRARY")
    media_dir = os.path.join(library_dir, "media")
    image_dir = os.path.join(library_dir, "image")
    sprite_dir = os.path.join(library_dir, "sprite")
    label_dir = os.path.join(library_dir, "label")
    
    for d in [media_dir, image_dir, sprite_dir, label_dir]:
        os.makedirs(d, exist_ok=True)
        
    with open(os.path.join(output_xfl_dir, "main.xfl"), "w", encoding="utf-8") as f:
        f.write("PROXY-CS5")
        
    media_elements = []
    symbol_includes = []

    for item in plist:
        name = item["name"]
        
        src_png = os.path.join(input_png_dir, f"{name}.png")
        dst_png = os.path.join(media_dir, f"{name}.png")
        if os.path.exists(src_png):
            shutil.copy(src_png, dst_png)
            
        bitmap_id = gen_uuid()
        media_elements.append(
            f'<DOMBitmapItem name="media/{name}" itemID="{bitmap_id}" sourceExternalFilepath="./LIBRARY/media/{name}.png" '
            f'sourceLastImported="0" allowSmoothing="true" useImportedJPEGData="false" compressionType="lossless" '
            f'originalCompressionType="lossless" href="media/{name}.png"/>'
        )
            
        scale_x = item.get("scale_x", 1.0)
        scale_y = item.get("scale_y", 1.0)
        origin_x = item.get("origin_x", 0.0)
        origin_y = item.get("origin_y", 0.0)
        
        image_id = gen_uuid()
        image_xml = f'''<DOMSymbolItem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://ns.adobe.com/xfl/2008/" name="image/{name}" itemID="{image_id}" symbolType="graphic">
  <timeline>
    <DOMTimeline name="{name}">
      <layers>
        <DOMLayer name="Layer 1" color="#4F80FF" current="true" isSelected="true">
          <frames>
            <DOMFrame index="0" keyMode="9728">
              <elements>
                <DOMBitmapInstance libraryItemName="media/{name}">
                  <matrix>
                    <Matrix a="{scale_x}" d="{scale_y}" tx="{origin_x}" ty="{origin_y}"/>
                  </matrix>
                </DOMBitmapInstance>
              </elements>
            </DOMFrame>
          </frames>
        </DOMLayer>
      </layers>
    </DOMTimeline>
  </timeline>
</DOMSymbolItem>'''
        with open(os.path.join(image_dir, f"{name}.xml"), "w", encoding="utf-8") as f:
            f.write(image_xml)
        symbol_includes.append(f'<Include href="image/{name}.xml" loadImmediate="false" itemID="{image_id}"/>')
            
        sprite_id = gen_uuid()
        sprite_xml = f'''<DOMSymbolItem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://ns.adobe.com/xfl/2008/" name="sprite/{name}" itemID="{sprite_id}" symbolType="graphic">
  <timeline>
    <DOMTimeline name="{name}">
      <layers>
        <DOMLayer name="1" color="#4F4FFF" current="true" isSelected="true">
          <frames>
            <DOMFrame index="0" keyMode="9728">
              <elements>
                <DOMSymbolInstance libraryItemName="image/{name}" symbolType="graphic" loop="loop"/>
              </elements>
            </DOMFrame>
          </frames>
        </DOMLayer>
      </layers>
    </DOMTimeline>
  </timeline>
</DOMSymbolItem>'''
        with open(os.path.join(sprite_dir, f"{name}.xml"), "w", encoding="utf-8") as f:
            f.write(sprite_xml)
        symbol_includes.append(f'<Include href="sprite/{name}.xml" loadImmediate="false" itemID="{sprite_id}"/>')

    def build_timeline_xml(folder_name, item_name, frames_list, x_offset=0.0, y_offset=0.0):
        symbol_name = f"{folder_name}/{item_name}"
        item_id = gen_uuid()
        
        flattened_frames = []
        for frame in frames_list:
            flat_children = flatten_frame_children(frame.get("children", []))
            flattened_frames.append({"children": flat_children})
            
        max_children = max((len(f.get("children", [])) for f in flattened_frames), default=0)
        
        layers_xml = ""
        
        if args.separate_layers:
            for layer_idx in range(max_children - 1, -1, -1):
                frames_xml = ""
                for frame_idx, frame in enumerate(flattened_frames):
                    children = frame.get("children", [])
                    if layer_idx < len(children):
                        child = children[layer_idx]
                        
                        if child.get("references_shared_animation"):
                            lib_item = f'sprite/{child["references_shared_animation"]}'
                        else:
                            original_name = child.get("name")
                            mapped_name = alias_map.get(original_name, original_name)
                            lib_item = f'sprite/{mapped_name}'
                            
                        mat_xml = matrix_to_xml(child.get("matrix", {}), x_offset, y_offset)
                        col_xml = color_to_xml(child.get("color", {}))
                        
                        frames_xml += f'''
            <DOMFrame index="{frame_idx}" duration="1" keyMode="9728">
              <elements>
                <DOMSymbolInstance libraryItemName="{lib_item}" symbolType="graphic" loop="loop">
                  <matrix>
                    {mat_xml}
                  </matrix>
                  {col_xml}
                </DOMSymbolInstance>
              </elements>
            </DOMFrame>'''
                    else:
                        frames_xml += f'\n            <DOMFrame index="{frame_idx}" duration="1" keyMode="9728"><elements/></DOMFrame>'
                        
                layers_xml += f'''
        <DOMLayer name="Layer_{layer_idx}" color="#000000">
          <frames>{frames_xml}
          </frames>
        </DOMLayer>'''
        else:
            frames_xml = ""
            for frame_idx, frame in enumerate(flattened_frames):
                children = frame.get("children", [])
                elements_xml = ""
                
                for child in children:
                    if child.get("references_shared_animation"):
                        lib_item = f'sprite/{child["references_shared_animation"]}'
                    else:
                        original_name = child.get("name")
                        mapped_name = alias_map.get(original_name, original_name)
                        lib_item = f'sprite/{mapped_name}'
                        
                    mat_xml = matrix_to_xml(child.get("matrix", {}), x_offset, y_offset)
                    col_xml = color_to_xml(child.get("color", {}))
                    
                    elements_xml += f'''
                <DOMSymbolInstance libraryItemName="{lib_item}" symbolType="graphic" loop="loop">
                  <matrix>
                    {mat_xml}
                  </matrix>
                  {col_xml}
                </DOMSymbolInstance>'''
                
                if elements_xml:
                    frames_xml += f'''
            <DOMFrame index="{frame_idx}" duration="1" keyMode="9728">
              <elements>{elements_xml}
              </elements>
            </DOMFrame>'''
                else:
                    frames_xml += f'\n            <DOMFrame index="{frame_idx}" duration="1" keyMode="9728"><elements/></DOMFrame>'
            
            layers_xml = f'''
        <DOMLayer name="Merged_Sprites" color="#000000">
          <frames>{frames_xml}
          </frames>
        </DOMLayer>'''
        
        # Fallback if animation is entirely empty
        if not layers_xml.strip():
            layers_xml = f'''
        <DOMLayer name="Layer_0" color="#000000">
          <frames>
            <DOMFrame index="0" duration="1" keyMode="9728"><elements/></DOMFrame>
          </frames>
        </DOMLayer>'''
    
        anim_xml = f'''<DOMSymbolItem xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://ns.adobe.com/xfl/2008/" name="{symbol_name}" itemID="{item_id}" symbolType="graphic">
  <timeline>
    <DOMTimeline name="{item_name}">
      <layers>{layers_xml}
      </layers>
    </DOMTimeline>
  </timeline>
</DOMSymbolItem>'''
        
        out_file = os.path.join(library_dir, folder_name, f"{item_name}.xml")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(anim_xml)
            
        return f'<Include href="{folder_name}/{item_name}.xml" loadImmediate="false" itemID="{item_id}"/>'

    for shared_name, shared_frames in shared_animations.items():
        symbol_includes.append(build_timeline_xml("sprite", shared_name, shared_frames))

    for anim in animations:
        anim_frames = frames[anim['start']:anim['start']+anim['duration']]
        symbol_includes.append(build_timeline_xml("label", anim['name'], anim_frames, x_offset=shift_x, y_offset=shift_y))

    labels_frames_xml = ""
    actions_frames_xml = ""
    instances_frames_xml = ""
    
    current_frame = 0
    for anim in animations:
        name = anim['name']
        dur = anim['duration']
        
        labels_frames_xml += f'\n            <DOMFrame index="{current_frame}" duration="{dur}" name="{name}" labelType="name"/>'
        
        if dur > 1:
            actions_frames_xml += f'\n            <DOMFrame index="{current_frame}" duration="{dur-1}" keyMode="9728"/>'
            actions_frames_xml += f'\n            <DOMFrame index="{current_frame + dur - 1}" duration="1" keyMode="9728"><Actionscript><script><![CDATA[stop();]]></script></Actionscript></DOMFrame>'
        else:
            actions_frames_xml += f'\n            <DOMFrame index="{current_frame}" duration="1" keyMode="9728"><Actionscript><script><![CDATA[stop();]]></script></Actionscript></DOMFrame>'
            
        instances_frames_xml += f'''
            <DOMFrame index="{current_frame}" duration="{dur}" keyMode="9728">
              <elements>
                <DOMSymbolInstance libraryItemName="label/{name}" symbolType="graphic" loop="loop">
                  <matrix>
                    <Matrix tx="0" ty="0"/>
                  </matrix>
                </DOMSymbolInstance>
              </elements>
            </DOMFrame>'''
        
        current_frame += dur
        
    if not labels_frames_xml:
        labels_frames_xml = '<DOMFrame index="0" duration="1" keyMode="9728"/>'
        actions_frames_xml = '<DOMFrame index="0" duration="1" keyMode="9728"/>'
        instances_frames_xml = '<DOMFrame index="0" duration="1" keyMode="9728"/>'

    dom_xml = f'''<DOMDocument xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://ns.adobe.com/xfl/2008/" width="390" height="390" frameRate="30" currentTimeline="1" xflVersion="2.96" creatorInfo="Generated by Python" platformId="0" platformInfo="ActionScript 3.0" id="{gen_uuid()}" vanishingPoint3DX="195" vanishingPoint3DY="195">
  <folders>
    <DOMFolderItem name="media" itemID="{gen_uuid()}"/>
    <DOMFolderItem name="image" itemID="{gen_uuid()}"/>
    <DOMFolderItem name="sprite" itemID="{gen_uuid()}"/>
    <DOMFolderItem name="label" itemID="{gen_uuid()}"/>
  </folders>
  <media>
    {chr(10).join("    " + element for element in media_elements)}
  </media>
  <symbols>
    {chr(10).join("    " + inc for inc in symbol_includes)}
  </symbols>
  <timelines>
    <DOMTimeline name="Scene 1">
      <layers>
        <DOMLayer name="labels" color="#FF0000" current="true" isSelected="true">
          <frames>{labels_frames_xml}
          </frames>
        </DOMLayer>
        <DOMLayer name="actions" color="#00FF00">
          <frames>{actions_frames_xml}
          </frames>
        </DOMLayer>
        <DOMLayer name="instances" color="#0000FF">
          <frames>{instances_frames_xml}
          </frames>
        </DOMLayer>
      </layers>
    </DOMTimeline>
  </timelines>
</DOMDocument>'''
    with open(os.path.join(output_xfl_dir, "DOMDocument.xml"), "w", encoding="utf-8") as f:
        f.write(dom_xml)

def main():
    parser = argparse.ArgumentParser(description="Convert BBONE to Adobe Animate XFL")
    parser.add_argument("input_file", help="Path to the .bbone file")
    parser.add_argument("--separate-layers", action="store_true", help="Generate a separate layer for each sprite (Warning: may cause lag)")
    parser.add_argument("--merge-similar", action="store_true", help="Detect and merge visually similar sprites to optimize performance")
    args = parser.parse_args()

    file_path = args.input_file
    path_to_file = Path(file_path)
    
    if not path_to_file.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    object_name = path_to_file.stem
    parent_dir = path_to_file.parent
    
    temp_dir = parent_dir / f"{object_name}_temp"
    output_xfl_dir = parent_dir / f"{object_name}_XFL"

    print(f"--- Starting Pipeline for {object_name} ---")

    print("[1/5] Unzipping BBONE data...")
    try:
        _, version, plugin_contents = unzip_bbone(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to unzip BBONE: {e}")
        sys.exit(1)

    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(exist_ok=True)

    json_data = {}

    print("[2/5] Parsing Atlas and Slicing Sprites...")
    try:
        plist = process_plugin_1(plugin_contents.get(1, b""), str(temp_dir))
        
        alias_map = {}
        unique_sprites = []
        
        if args.merge_similar and plist:
            print("-> Analyzing sprites for visual duplicates to optimize performance...")
            new_plist = []
            for item in plist:
                name = item["name"]
                img_path = os.path.join(temp_dir, f"{name}.png")
                
                if not os.path.exists(img_path):
                    new_plist.append(item)
                    continue

                try:
                    with Image.open(img_path) as img:
                        img_rgba = img.convert("RGBA")
                        w, h = img_rgba.size
                        fp = np.array(img_rgba.resize((16, 16), RESAMPLE_FILTER), dtype=np.float32)
                except Exception:
                    new_plist.append(item)
                    continue

                matched = False
                for u in unique_sprites:
                    if abs(u["w"] - w) <= 3 and abs(u["h"] - h) <= 3:
                        mse = np.mean((u["fp"] - fp) ** 2)
                        if mse < 10.0:
                            alias_map[name] = u["name"]
                            matched = True
                            try:
                                os.remove(img_path)
                            except Exception: 
                                pass
                            break
                
                if not matched:
                    unique_sprites.append({"name": name, "fp": fp, "w": w, "h": h})
                    new_plist.append(item)

            print(f"-> Merged {len(plist) - len(new_plist)} duplicate sprites.")
            json_data["plist"] = new_plist
        else:
            json_data["plist"] = plist
            
        json_data["alias_map"] = alias_map

    except Exception as e:
        print(f"[ERROR] Failed to process atlas: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("[3/5] Extracting Animation Timelines and Labels...")
    try:
        labels = parse_frame_labels(plugin_contents.get(3, b""))
        json_data["labels"] = labels
        animation_json = decode_animation_chunk(plugin_contents.get(2, b""), version)
        json_data["animation"] = animation_json
    except Exception as e:
        print(f"[ERROR] Failed to extract animation data: {e}")
        sys.exit(1)

    print("[4/5] Building Adobe Animate XFL Project...")
    try:
        build_xfl(json_data, str(temp_dir), str(output_xfl_dir), args)
    except Exception as e:
        print(f"[ERROR] XFL Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("[5/5] Cleaning up temporary files...")
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        print(f"-> [WARNING] Could not fully delete temporary directory: {e}")

    print(f"\n-> Success! Pipeline Complete. Output saved to: {output_xfl_dir}")

if __name__ == "__main__":
    main()