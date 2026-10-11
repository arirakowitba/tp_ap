"""Probe public picture URLs before attempting an image-model experiment."""
from pathlib import Path
import argparse,json,urllib.request,urllib.error,time

def probe(data_dir,output,limit=5):
    records=[];urls=[]
    with (Path(data_dir)/'train_data.jsonlines').open() as f:
        for line in f:
            row=json.loads(line)
            for picture in row.get('pictures') or []:
                url=picture.get('secure_url') or picture.get('url')
                if url and url not in urls:urls.append(url)
                if len(urls)>=limit:break
            if len(urls)>=limit:break
    for url in urls:
        variants=[url]
        if url.startswith('http://'):variants.append('https://'+url[7:])
        for variant in variants:
            started=time.time()
            try:
                request=urllib.request.Request(variant,headers={'User-Agent':'Mozilla/5.0'})
                with urllib.request.urlopen(request,timeout=15) as response:
                    status=response.status;content_type=response.headers.get('Content-Type');sample=response.read(1024)
                available=status==200 and content_type and content_type.startswith('image/') and len(sample)>0
                record=dict(url=variant,status=status,content_type=content_type,available=bool(available))
            except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError) as exc:
                record=dict(url=variant,status=getattr(exc,'code',None),available=False,error=str(exc))
            record['seconds']=time.time()-started;records.append(record)
    result=dict(probes=records,accessible=sum(r['available'] for r in records),image_model_trained=False,
                note='Image embeddings require accessible original pictures; URL access alone is not an image experiment.')
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=5);args=p.parse_args();probe(args.data_dir,args.output,args.limit)
