var Highlighter = /** @class */ (function () {
    function Highlighter() {
        var _this = this;
        this.onClick = function (e) {
            var t = e.currentTarget.textContent;
            if (!(t in _this.nodes)) {
                console.log(e.currentTarget);
                return;
            }
            if (_this.highlighted == t) {
                _this.unhighlight();
            }
            else {
                if (_this.highlighted != '') {
                    _this.unhighlight();
                }
                _this.highlightWord(t);
            }
        };
        this.nodes = {};
        this.highlighted = '';
        this.addNodes(document.querySelectorAll('.highlight .n'));
        this.addNodes(document.querySelectorAll('.highlight .nn'));
        this.addNodes(document.querySelectorAll('.highlight .nc'));
        this.addNodes(document.querySelectorAll('.highlight .nf'));
        for (var k in this.nodes) {
            for (var _i = 0, _a = this.nodes[k]; _i < _a.length; _i++) {
                var n = _a[_i];
                n.addEventListener('click', this.onClick);
            }
        }
    }
    Highlighter.prototype.unhighlight = function () {
        for (var _i = 0, _a = this.nodes[this.highlighted]; _i < _a.length; _i++) {
            var n = _a[_i];
            n.classList.remove('clicked');
        }
        this.highlighted = '';
        document.body.classList.remove('lights-off');
    };
    Highlighter.prototype.highlightWord = function (t) {
        for (var _i = 0, _a = this.nodes[t]; _i < _a.length; _i++) {
            var n = _a[_i];
            n.classList.add('clicked');
        }
        document.body.classList.add('lights-off');
        this.highlighted = t;
    };
    Highlighter.prototype.addNodes = function (nodes) {
        for (var i = 0; i < nodes.length; ++i) {
            var name_1 = nodes[i].textContent;
            if (!(name_1 in this.nodes)) {
                this.nodes[name_1] = [];
            }
            this.nodes[name_1].push(nodes[i]);
        }
    };
    return Highlighter;
}());
var KatexHighlighter = /** @class */ (function () {
    function KatexHighlighter() {
        var _this = this;
        this.onClick = function (e) {
            e.preventDefault();
            e.stopPropagation();
            var t = KatexHighlighter.getName(e.currentTarget);
            // console.log('clicked', t)
            if (!(t in _this.nodes)) {
                console.log(e.currentTarget);
                return;
            }
            if (_this.highlighted == t) {
                _this.unhighlight();
            }
            else {
                if (_this.highlighted != '') {
                    _this.unhighlight();
                }
                _this.highlightWord(t);
            }
        };
        this.nodes = {};
        this.highlighted = '';
        this.addNodes(document.querySelectorAll('.coloredeq'));
        for (var k in this.nodes) {
            for (var _i = 0, _a = this.nodes[k]; _i < _a.length; _i++) {
                var n = _a[_i];
                n.addEventListener('click', this.onClick);
            }
        }
    }
    KatexHighlighter.prototype.unhighlight = function () {
        for (var _i = 0, _a = this.nodes[this.highlighted]; _i < _a.length; _i++) {
            var n = _a[_i];
            n.classList.remove('clicked');
        }
        this.highlighted = '';
        document.body.classList.remove('lights-off');
    };
    KatexHighlighter.prototype.highlightWord = function (t) {
        for (var _i = 0, _a = this.nodes[t]; _i < _a.length; _i++) {
            var n = _a[_i];
            n.classList.add('clicked');
        }
        document.body.classList.add('lights-off');
        this.highlighted = t;
    };
    KatexHighlighter.getName = function (node) {
        var name = '';
        for (var i = 0; i < node.classList.length; ++i) {
            var cn = node.classList[i];
            if (cn.substr(0, 2) === 'eq') {
                name = cn.substr(2);
            }
        }
        return name;
    };
    KatexHighlighter.prototype.addNodes = function (nodes) {
        for (var i = 0; i < nodes.length; ++i) {
            var name_2 = KatexHighlighter.getName(nodes[i]);
            console.log(nodes[i]);
            if (!(name_2 in this.nodes)) {
                this.nodes[name_2] = [];
            }
            this.nodes[name_2].push(nodes[i]);
        }
    };
    return KatexHighlighter;
}());
function addListeners() {
    var h = new Highlighter();
    var k = new KatexHighlighter();
}
if (document.readyState == "complete") {
    addListeners();
}
else {
    document.onreadystatechange = function () {
        if (document.readyState == "complete") {
            addListeners();
        }
    };
}
