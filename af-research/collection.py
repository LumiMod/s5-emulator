#!/usr/bin/env python3
"""Stock-camera collection folders, streaming hashes and private handoff ZIPs.

Python 3.10+, standard library only. Does not control or modify the camera,
label images, infer lens encoder positions, or upload any files.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile

SEQUENCES = [
    ('S01_static_normal', 'static_acquisition', 'normal', 'static', 'AFS'),
    ('S02_static_low', 'static_acquisition', 'low', 'static', 'AFS'),
    ('S03_approach_normal', 'approach_tracking', 'normal', 'moving', 'AFC'),
    ('S04_approach_low', 'approach_tracking', 'low', 'moving', 'AFC'),
    ('S05_lateral_occlusion', 'lateral_occlusion', 'normal', 'moving', 'AFC'),
    ('S06_manual_reference', 'manual_reference', 'unknown', 'static', 'MF'),
]
MEDIA = {'.jpg','.jpeg','.rw2','.mov','.mp4','.png','.wav','.m4v'}
CONFIGS = ('session.json','sequences.json','events.csv')
EVENT_FIELDS = ['sequence_id','trial_id','phone_file','phone_time_s','camera_file',
                'camera_time_s','event','observation','notes']


def need(ok, message):
    if not ok:
        raise ValueError(message)


def write_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    need(isinstance(relative,str) and relative and '\\' not in relative,'路径必须是非空的相对 POSIX 路径')
    p=Path(relative)
    need(not p.is_absolute() and '..' not in p.parts,'路径必须在采集目录内：'+str(p))
    need(p.parts and p.as_posix()==relative,'路径必须采用规范形式：'+relative)
    target=root/p
    current=root
    for part in p.parts:
        current=current/part
        need(not current.is_symlink(),'不接受符号链接：'+str(p))
    need(target.resolve().is_relative_to(root.resolve()),'路径越界：'+str(p))
    return target


def initialize(root, session_id):
    need(re.fullmatch(r'[A-Za-z0-9_-]{1,80}',session_id),'session-id 仅限字母、数字、_、-')
    need(not root.exists(),'目标已存在；请使用新目录，避免覆盖已有采集')
    root.mkdir(parents=True)
    write_json(root/'session.json',{
        'schema_version':1,'session_id':session_id,'owner_alias':None,
        'camera_model':'DC-S5','camera_firmware':None,'lens_model':None,'lens_firmware':None,
        'capture_date':None,'capture_timezone':None,'notes':'未知项目填 null；不要估造照度或镜头编码器位置。',
        'participants_agreed_to_private_research':None})
    rows=[]
    for sid,purpose,light,motion,mode in SEQUENCES:
        for kind in ('camera','phone'):(root/'raw'/sid/kind).mkdir(parents=True)
        rows.append({'sequence_id':sid,'purpose':purpose,'light':light,'motion':motion,
            'af_mode':mode,'af_area':None,'subject_detection':None,'focal_length_mm':None,
            'aperture':None,'shutter':None,'iso':None,'video_resolution':None,'video_fps':None,
            'video_continuous_af':None,'stabilization':None,'af_custom_settings':None,
            'illuminance_lux':None,'distance_m_approx':None,'planned_trials':None,
            'completed_trials':None,'notes':(
                '本组包含正常光与弱光参考；请在记录表按文件名区分 normal/low，研究者正式标注时拆成独立序列。'
                if sid=='S06_manual_reference' else '以上模式是试采计划，拍摄后按实际设置修正。')})
    write_json(root/'sequences.json',{'schema_version':1,'sequences':rows})
    with (root/'events.csv').open('w',encoding='utf-8-sig',newline='') as f:
        csv.writer(f).writerow(EVENT_FIELDS)
    (root/'settings').mkdir()
    print('已创建',root)


def inventory(root):
    need(root.is_dir() and not root.is_symlink(),'采集目录无效')
    paths=[safe_path(root,name) for name in CONFIGS]
    for directory in ('raw','settings'):
        top=safe_path(root,directory)
        need(top.is_dir(),'缺少目录：'+directory)
        for path in sorted(top.rglob('*')):
            need(not path.is_symlink(),'不接受符号链接：'+str(path.relative_to(root)))
            if path.is_file() and path.name!='.DS_Store':paths.append(path)
    rows=[]
    for path in paths:
        need(path.is_file(),'缺少文件：'+str(path.relative_to(root)))
        rows.append({'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    report={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
            'files':rows,'total_bytes':sum(r['bytes'] for r in rows),
            'scope':'File inventory and integrity only; no automatic annotation, camera authenticity or AF quality claim.'}
    write_json(safe_path(root,'collection-manifest.json'),report)
    return report


def validate(root, *, manifest_bytes=None):
    need(root.is_dir() and not root.is_symlink(),'采集目录无效')
    session=json.loads(safe_path(root,'session.json').read_text(encoding='utf-8'))
    document=json.loads(safe_path(root,'sequences.json').read_text(encoding='utf-8'))
    need(isinstance(session,dict) and isinstance(document,dict),'元数据必须是 JSON 对象')
    need(session.get('schema_version')==document.get('schema_version')==1,'不支持的 schema_version')
    need(session.get('camera_model')=='DC-S5','此采集项目针对第一代 DC-S5；请核实机型')
    session_id=session.get('session_id')
    need(isinstance(session_id,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}',session_id),'session_id 无效')
    sequences=document.get('sequences');need(isinstance(sequences,list) and sequences,'缺少场景清单')
    ids=set();warnings=[]
    for row in sequences:
        need(isinstance(row,dict),'场景条目必须是 JSON 对象')
        sid=row.get('sequence_id')
        need(isinstance(sid,str) and re.fullmatch(r'[A-Za-z0-9_-]+',sid) and sid not in ids,'重复/非法场景编号')
        need(row.get('light') in ('normal','low','backlit','unknown',None,''),'非法光线标签')
        need(row.get('motion') in ('static','moving','unknown',None,''),'非法运动标签')
        ids.add(sid)
    for field in ['camera_firmware','lens_model','capture_date','capture_timezone']:
        if not session.get(field):warnings.append('待补充 '+field)
    if session.get('participants_agreed_to_private_research') is not True:
        warnings.append('分享前确认参与者同意此用途；素材保留在私下交接渠道')
    if manifest_bytes is None:
        manifest_bytes=safe_path(root,'collection-manifest.json').read_bytes()
    manifest=json.loads(manifest_bytes)
    need(isinstance(manifest,dict) and manifest.get('schema_version')==1,'清单版本无效')
    need(isinstance(manifest.get('files'),list),'清单缺少文件列表')
    seen=set();media_counts={sid:{'camera':0,'phone':0} for sid in ids}
    for entry in manifest['files']:
        need(isinstance(entry,dict),'清单文件条目无效')
        name=entry.get('path')
        path=safe_path(root,name)
        need(name not in seen,'清单路径重复');seen.add(name)
        parts=Path(name).parts
        need(name in CONFIGS or (len(parts)>1 and parts[0] in ('raw','settings')),'清单包含范围外文件：'+name)
        need(type(entry.get('bytes')) is int and entry['bytes']>=0,'清单文件大小无效：'+name)
        digest=entry.get('sha256')
        need(isinstance(digest,str) and re.fullmatch(r'[0-9a-f]{64}',digest),'清单 SHA256 无效：'+name)
        need(path.is_file() and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'文件改变/缺失：'+name)
        if parts[0]=='raw':
            need(len(parts)>=4 and parts[1] in ids and parts[2] in ('camera','phone'),'素材目录不符合场景结构：'+name)
            if path.suffix.lower() in MEDIA:
                need(entry['bytes']>0,'素材文件为空：'+name)
                media_counts[parts[1]][parts[2]]+=1
            else:warnings.append('未识别素材类型，仍保留：'+name)
    need(type(manifest.get('total_bytes')) is int and manifest['total_bytes']==sum(e['bytes'] for e in manifest['files']),
         '清单总字节数不匹配')
    need(set(CONFIGS)<=seen,'清单缺少元数据文件')
    for dirname in ('raw','settings'):
        directory=safe_path(root,dirname)
        need(directory.is_dir(),'缺少目录：'+dirname)
        for path in directory.rglob('*'):
            need(not path.is_symlink(),'不接受符号链接')
            if path.is_file() and path.name!='.DS_Store':
                need(path.relative_to(root).as_posix() in seen,'有新增文件，请重新 inventory：'+path.name)
    with safe_path(root,'events.csv').open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        need(reader.fieldnames==EVENT_FIELDS,'events.csv 列名不匹配')
        events=list(reader)
    for row in events:
        need(None not in row and all(value is not None for value in row.values()),'事件行的字段数量不匹配')
        need(row['sequence_id'] in ids,'事件的场景编号不存在')
        for field in ('phone_time_s','camera_time_s'):
            if row[field]:
                value=float(row[field]);need(value>=0 and math.isfinite(value),'事件时间应是从文件起点计的非负秒数')
                need(row[field.replace('_time_s','_file')],'事件时间必须对应一个文件：'+field)
        for field,kind in [('phone_file','phone'),('camera_file','camera')]:
            if row[field]:
                rel='raw/'+row['sequence_id']+'/'+kind+'/'+row[field]
                need(safe_path(root,rel).is_file() and rel in seen,'事件引用文件不存在：'+rel)
    total=sum(sum(v.values()) for v in media_counts.values())
    for sid,counts in sorted(media_counts.items()):
        if not counts['camera']:warnings.append(sid+' 尚无相机原始文件')
        if not counts['phone']:warnings.append(sid+' 尚无手机拍屏；无法从原生视频推断对焦框')
    if not events:warnings.append('没有事件记录；可先私下提交试采，由研究者协助整理')
    return {'structural_and_hash_checks_passed':True,'media_files':total,'sequence_files':media_counts,
            'events':len(events),'warnings':warnings,'ready_for_research_review':total>0,
            'ready_for_training':False,'raw_media_decoding_verified':False,
            'limits':'This is an integrity check, not image annotation, internal-DSP capture or a focus-ground-truth measurement.'}


def package(root,destination):
    root=root.absolute()
    need(not destination.resolve().is_relative_to(root.resolve()),'ZIP 应放在采集目录之外')
    checksum_path=destination.with_name(destination.name+'.sha256')
    need(not destination.exists() and not destination.is_symlink(),'ZIP 已存在，请换名字以免覆盖')
    need(not checksum_path.exists() and not checksum_path.is_symlink(),'校验文件已存在，请换名字以免覆盖')
    # Validate and archive the same immutable manifest snapshot.
    manifest_bytes=safe_path(root,'collection-manifest.json').read_bytes()
    report=validate(root,manifest_bytes=manifest_bytes);need(report['media_files']>0,'没有素材，不能打包空模板')
    manifest=json.loads(manifest_bytes)
    archive_prefix=root.resolve().name
    need(archive_prefix and '\\' not in archive_prefix,'采集目录名不适合跨平台 ZIP 路径')
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation avoids overwriting either artifact, including a concurrent writer.
    created_zip=False;created_checksum=False
    try:
        with destination.open('xb') as output:
            created_zip=True
            with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
                for entry in manifest['files']:
                    archive.write(safe_path(root,entry['path']),arcname=archive_prefix+'/'+entry['path'])
                archive.writestr(archive_prefix+'/collection-manifest.json',manifest_bytes)
        # Verify actual archived bytes, including files changed during packaging.
        with zipfile.ZipFile(destination) as archive:
            need(archive.read(archive_prefix+'/collection-manifest.json')==manifest_bytes,'ZIP 清单校验失败')
            for entry in manifest['files']:
                h=hashlib.sha256();size=0
                with archive.open(archive_prefix+'/'+entry['path']) as stream:
                    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block);size+=len(block)
                need(size==entry['bytes'] and h.hexdigest()==entry['sha256'],'ZIP 内容校验失败')
        checksum=sha(destination)
        with checksum_path.open('x',encoding='utf-8') as output:
            created_checksum=True
            output.write(checksum+'  '+destination.name+'\n')
    except BaseException:
        if created_checksum:checksum_path.unlink(missing_ok=True)
        if created_zip:destination.unlink(missing_ok=True)
        raise
    print('已打包并校验',destination,'SHA256',checksum)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('init');p.add_argument('directory',type=Path);p.add_argument('--session-id',required=True)
    for action in ('inventory','check'):
        p=sub.add_parser(action);p.add_argument('directory',type=Path)
    p=sub.add_parser('pack');p.add_argument('directory',type=Path);p.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    try:
        if args.command=='init':initialize(args.directory,args.session_id)
        elif args.command=='inventory':
            r=inventory(args.directory);print('已记录',len(r['files']),'个文件，',r['total_bytes'],'字节')
        elif args.command=='check':print(json.dumps(validate(args.directory),ensure_ascii=False,indent=2))
        else:package(args.directory,args.output)
    except (ValueError,KeyError,TypeError,OSError,zipfile.BadZipFile) as exc:
        print('检查失败：'+str(exc),file=sys.stderr);raise SystemExit(2)


if __name__=='__main__':main()
