'use strict';

var App = (function() {
  function App(options) {
    var defaults = {};
    this.opt = $.extend({}, defaults, options);
    this.init();
  }

  App.prototype.init = function(){
    var _this = this;

    $.when($.getJSON(this.opt.geojson)).done(function(data) {
      _this.onReady(data);
    });

    this.colorKey = new ColorKey(this.opt.colorKey);
  };

  App.prototype.loadListeners = function(){
    var _this = this;

    var globes = this.globes;

    $(window).on('resize', function(){
      _.each(globes, function(globe){
        globe.onResize();
      });
    });
  };

  App.prototype.onReady = function(geojson){
    console.log("All globe data loaded");

    $('.loading').remove();

    var globes = [];
    var globesOpt = this.opt.globes;

    _.each(globesOpt, function(opt){
      globes.push(new Globe(_.extend({}, opt, {"geojson": geojson})));
    });

    this.globes = globes;
    this.calendar = new Calendar(_.extend({}, this.opt.calendar));

    this.loadListeners();

    this.render();
  };

  App.prototype.render = function(){
    var _this = this;

    var globeTest = this.globes[0];
    var yearProgress = globeTest.getProgress();

    _.each(this.globes, function(globe){
      globe.render();
    });

    this.calendar.render(yearProgress);

    requestAnimationFrame(function(){ _this.render(); });
  };

  return App;

})();

$(function() {
  var app = new App(CONFIG);
});
