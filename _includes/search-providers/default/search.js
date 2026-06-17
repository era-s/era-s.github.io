var SOURCES = window.TEXT_VARIABLES.sources;
var PAHTS = window.TEXT_VARIABLES.paths;
window.Lazyload.js([SOURCES.jquery, PAHTS.search_js], function() {
  var search = (window.search || (window.search = {}));
  var searchData = window.TEXT_SEARCH_DATA || {};

  // Max results shown per collection group, and how much context to keep
  // around a content match when building the result snippet.
  var MAX_PER_GROUP = 10;
  var SNIPPET_RADIUS = 60;

  function memorize(f) {
    var cache = {};
    return function () {
      var key = Array.prototype.join.call(arguments, ',');
      if (key in cache) return cache[key];
      else return cache[key] = f.apply(this, arguments);
    };
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function escapeRegExp(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
  // Escape text for safe HTML, then wrap each case-insensitive occurrence of
  // the query in <mark> so matches are highlighted.
  function highlight(text, query) {
    if (!text) { return ''; }
    var escaped = escapeHtml(text);
    if (!query) { return escaped; }
    var re = new RegExp('(' + escapeRegExp(escapeHtml(query)) + ')', 'gi');
    return escaped.replace(re, '<mark>$1</mark>');
  }
  // Build a short, highlighted snippet of the body around the first match.
  function makeSnippet(content, query) {
    if (!content) { return ''; }
    var idx = content.toLowerCase().indexOf(query.toLowerCase());
    if (idx < 0) { return ''; }
    var start = Math.max(0, idx - SNIPPET_RADIUS);
    var end = Math.min(content.length, idx + query.length + SNIPPET_RADIUS);
    var snippet = (start > 0 ? '…' : '') + content.slice(start, end) +
      (end < content.length ? '…' : '');
    return highlight(snippet, query);
  }

  /// search — matches against title, tags and body content
  function searchByQuery(query) {
    var i, j, key, keys, cur, result = {};
    var q = query.toLowerCase();
    keys = Object.keys(searchData);
    for (i = 0; i < keys.length; i++) {
      key = keys[i];
      for (j = 0; j < searchData[key].length; j++) {
        if (result[key] && result[key].length >= MAX_PER_GROUP) { break; }
        cur = searchData[key][j];
        var title = cur.title || '', tags = cur.tags || '', content = cur.content || '';
        var inTitle = title.toLowerCase().indexOf(q) >= 0;
        var inTags = tags.toLowerCase().indexOf(q) >= 0;
        var inContent = content.toLowerCase().indexOf(q) >= 0;
        if (inTitle || inTags || inContent) {
          if (result[key] === undefined) { result[key] = []; }
          var snippet = '';
          if (inContent) {
            snippet = makeSnippet(content, query);
          } else if (inTags) {
            snippet = '<i class="fas fa-tag"></i> ' + highlight(tags, query);
          }
          result[key].push({ title: cur.title, url: cur.url, snippet: snippet });
        }
      }
    }
    return result;
  }

  var renderHeader = memorize(function(header) {
    return $('<p class="search-result__header">' + header + '</p>');
  });

  var renderItem = function(index, item, query) {
    var titleHtml = highlight(item.title, query);
    var snippetHtml = item.snippet ?
      '<span class="search-result__snippet">' + item.snippet + '</span>' : '';
    return $('<li class="search-result__item" data-index="' + index + '">' +
      '<a class="button search-result__link" href="' + item.url + '">' +
      '<span class="search-result__title">' + titleHtml + '</span>' + snippetHtml +
      '</a></li>');
  };

  function render(data, query) {
    if (!data) { return null; }
    var $root = $('<ul></ul>'), i, j, key, keys, itemIndex = 0;
    keys = Object.keys(data);
    for (i = 0; i < keys.length; i++) {
      key = keys[i];
      $root.append(renderHeader(key));
      for (j = 0; j < data[key].length; j++) {
        $root.append(renderItem(itemIndex++, data[key][j], query));
      }
    }
    return $root;
  }

  // search box
  var $result = $('.js-search-result'), $resultItems;
  var lastActiveIndex, activeIndex;

  function clear() {
    $result.html(null);
    $resultItems = $('.search-result__item'); activeIndex = 0;
  }
  function onInputNotEmpty(val) {
    $result.html(render(searchByQuery(val), val));
    $resultItems = $('.search-result__item'); activeIndex = 0;
    $resultItems.eq(0).addClass('active');
  }

  search.clear = clear;
  search.onInputNotEmpty = onInputNotEmpty;

  function updateResultItems() {
    lastActiveIndex >= 0 && $resultItems.eq(lastActiveIndex).removeClass('active');
    activeIndex >= 0 && $resultItems.eq(activeIndex).addClass('active');
  }

  function moveActiveIndex(direction) {
    var itemsCount = $resultItems ? $resultItems.length : 0;
    if (itemsCount > 1) {
      lastActiveIndex = activeIndex;
      if (direction === 'up') {
        activeIndex = (activeIndex - 1 + itemsCount) % itemsCount;
      } else if (direction === 'down') {
        activeIndex = (activeIndex + 1 + itemsCount) % itemsCount;
      }
      updateResultItems();
    }
  }

  // Char Code: 13  Enter, 37  ⬅, 38  ⬆, 39  ➡, 40  ⬇
  $(window).on('keyup', function(e) {
    var modalVisible = search.getModalVisible && search.getModalVisible();
    if (modalVisible) {
      if (e.which === 38) {
        modalVisible && moveActiveIndex('up');
      } else if (e.which === 40) {
        modalVisible && moveActiveIndex('down');
      } else if (e.which === 13) {
        modalVisible && $resultItems && activeIndex >= 0 && $resultItems.eq(activeIndex).children('a')[0].click();
      }
    }
  });

  $result.on('mouseover', '.search-result__item > a', function() {
    var itemIndex = $(this).parent().data('index');
    itemIndex >= 0 && (lastActiveIndex = activeIndex, activeIndex = itemIndex, updateResultItems());
  });
});
