
import io
import re
import requests
import sys
import tarfile

def get_arxiv_ids(url):
    html = requests.get(url).text
    urls = re.findall(r'https?://arxiv.org/abs/[^"\',]*', html)
    return {re.search(r'\d+\.[A-Za-z0-9]+$', url).group(0) for url in urls}

def fetch_tex_files(arxiv_ids):
    tex_files = []
    for arxiv_id in arxiv_ids:
        url = 'https://arxiv.org/e-print/{}'.format(arxiv_id)
        raw = requests.get(url).content
        try:
            tar = tarfile.open(fileobj=io.BytesIO(raw))
        except tarfile.ReadError as e:
            print('Couldnt untar file from url {}'.format(url))
            continue
        for file_ in tar.getmembers():
            if file_.name[-4:] == '.tex':
                tex_files.append(tar.extractfile(file_))
        print('Got TeX file from {}'.format(url))
    return tex_files

def main():
    url = ("http://www.iro.umontreal.ca/~lisa/publications2/"
           "index.php/publications/showlist/year/{}")
    urls = [url.format(i) for i in range(39)]
    urls.append('http://www.di.ens.fr/~slacoste/')
    arxiv_ids = set()
    for url in urls:
        print('Sniffing {} for arxiv IDs'.format(url))
        arxiv_ids = arxiv_ids.union(get_arxiv_ids(url))
    print('Found {} arxiv IDs'.format(len(arxiv_ids)))
    tex_files = fetch_tex_files(arxiv_ids)
    print('Found {} TeX files'.format(len(tex_files)))
    filepath = 'all_tex_files.txt'
    with open(filepath, 'w') as fh:
        for tex_file in tex_files:
            fh.write(tex_file.read().decode(errors='ignore'))
            fh.write('\n\n\n')
    print('Produced mega TeX/txt file in {}'.format(filepath))

if __name__ == '__main__':
    main()

