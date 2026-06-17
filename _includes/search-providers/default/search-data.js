window.TEXT_SEARCH_DATA={
  {%- for _collection in site.collections -%}
    {%- unless forloop.first -%},{%- endunless -%}
    '{{ _collection.label }}':[
      {%- for _article in _collection.docs -%}
      {%- unless forloop.first -%},{%- endunless -%}
      {%- include snippets/prepend-baseurl.html path=_article.url -%}
      {%- assign _url = __return -%}
      {'title':{{ _article.title | jsonify }},
      'url':{{ _url | jsonify }},
      'tags':{{ _article.tags | join: ', ' | jsonify }},
      'content':{{ _article.content | strip_html | strip_newlines | truncate: 2000 | jsonify }}}
      {%- endfor -%}
    ]
  {%- endfor -%}
};
