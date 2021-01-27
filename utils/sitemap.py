from pathlib import Path

import git

HOME = Path('./labml_nn')
REPO = git.Repo('.')


def collect(path: Path):
    if path.is_file():
        try:
            commit = next(iter(REPO.iter_commits(paths=path)))
        except StopIteration:
            return []

        html = path.relative_to(HOME)
        if html.stem == '__init__':
            html = html.parent / 'index.html'
        else:
            html = html.parent / f'{html.stem}.html'

        return [{'path': str(html), 'date': str(commit.committed_datetime.date())}]

    urls = []
    for f in path.iterdir():
        urls += collect(f)

    return urls


def main():
    urls = []
    for f in HOME.iterdir():
        urls += collect(f)

    urls = [f'''
    <url>
      <loc>https://nn.labml.ai/{u['path']}</loc>
      <lastmod>{u['date']}T16:30:00+00:00</lastmod>
      <priority>1.00</priority>
    </url>
    ''' for u in urls]

    urls = '\n'.join(urls)
    xml = f'''
    <?xml version="1.0" encoding="UTF-8"?>
    <urlset
      xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="http://www.sitemaps.org/schemas/sitemap/0.9
            http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd">
      {urls}
    </urlset>
    '''

    with open(str(HOME.parent / 'docs' / 'sitemap.xml'), 'w') as f:
        f.write(xml)


if __name__ == '__main__':
    main()
