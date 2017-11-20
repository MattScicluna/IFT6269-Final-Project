
import io
import re
import requests
import sys
import tarfile
import zipfile

def get_arxiv_ids(url):
    html = requests.get(url).text
    urls = re.findall(r'https?://arxiv.org/abs/[^"]*', html)
    return {re.search(r'\d+\.\d+$', url).group(0) for url in urls}

def fetch_tex_files(arxiv_ids):
    tex_files = []
    for arxiv_id in arxiv_ids:
        url = 'https://arxiv.org/e-print/{}'.format(arxiv_id)
        raw = requests.get(url).content
        tar = tarfile.open(fileobj=io.BytesIO(raw))
        for file_ in tar.getmembers():
            if file_.name[-4:] == '.tex':
                tex_files.append((file_.name, tar.extractfile(file_)))
    return tex_files

def main():
    url = 'http://www.di.ens.fr/~slacoste/'
    print('Sniffing {} for arxiv IDs'.format(url))
    arxiv_ids = get_arxiv_ids(url)
    print('Found {} arxiv IDs'.format(len(arxiv_ids)))
    tex_files = fetch_tex_files(arxiv_ids)
    print('Found {} TeX files'.format(len(tex_files)))
    filepath = 'simon.zip'
    zf = zipfile.ZipFile(filepath, mode='w')
    for tex_file_name, tex_file_content in tex_files:
        zf.writestr(tex_file_name, tex_file_content.read())
    zf.close()
    print('Archived TeX files in {}'.format(filepath))

if __name__ == '__main__':
    main()

