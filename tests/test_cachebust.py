"""The served index page must cache-bust app.js / styles.css (append ?v=<mtime>) so a code
change is picked up without a manual hard-refresh — the recurring 'I still see the old UI' bug."""
from webapp.server import _index_html


def test_index_html_versions_local_assets():
    html = _index_html()
    assert "/static/app.js?v=" in html
    assert "/static/styles.css?v=" in html


def test_no_unversioned_local_asset_urls_remain():
    html = _index_html()
    # every reference is the cache-busted form (so the browser can't reuse a stale copy)
    assert 'src="/static/app.js"' not in html
    assert 'href="/static/styles.css"' not in html
