import os
import json
import hashlib
import tarfile
import requests

from tqdm import tqdm

class Cholec80:
    def __init__(self, args):
        self.outfile = os.path.join(args.data_rootdir, "cholec80.tar.gz")
        self.outdir = os.path.join(args.data_rootdir, "cholec80")
        self.check_sum = args.verify_checksum

        self.URL = "https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.tar.gz"
        self.true_chk = 'aee27edbb4454d399e8dc8195d128833'
        self.CHUNK_SIZE = 2 ** 20


    def download_data(self):
        print(f'Downloading archive to {self.outdir}.')
        with requests.get(self.URL, stream=True) as r:
            r.raise_for_status()
            total_length = int(float(r.headers.get("content-length")) / 10 ** 6) 
            progress_bar = tqdm(unit="MB", total=total_length)

            with open(self.outfile, "wb") as f:
                for chunk in r.iter_content(chunk_size=self.CHUNK_SIZE):
                    progress_bar.update(len(chunk) / 10 ** 6)
                    f.write(chunk)


        if self.check_sum:
            self.verify_checksum()
        else:
            pass
        
        return
   
    def verify_checksum(self):
        print('Checking integrity of data.')
        
        m = hashlib.md5()

        with open(self.outfile, 'rb') as f:
            while True:
                data = f.read(self.CHUNK_SIZE)
                if not data:
                    break
                m.update(data)

        chk = m.hexdigest()

        print("Checksum: {}".format(chk))
        print(f"True Checksum: {self.true_chk}")
        assert m.hexdigest() == chk, 'Data did not download correctly.'

        return

    
    def file_extraction(self):
        print('Extracting files.')
        with tarfile.open(self.outfile, "r") as t:
            t.extractall(self.outdir) 

        self.config_setup()
        return
    
    
    def config_setup(self):
        with open("tf_cholec80/configs/config.json", "r") as f:
            config = json.loads(f.read())

        config["cholec80_dir"] = self.outdir
        json_string = json.dumps(config, indent=2, sort_keys=True)

        with open("tf_cholec80/configs/config.json", "w") as f:
            f.write(json_string)

        print("All done - config saved to {}".format(
        os.path.join(os.getcwd(), "config.json"))
        )


