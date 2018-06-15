<template>
  <div class="parent app-container">
    <div class="child1" style="width:550px;height:550px">
      <p>draw a digit here!</p>
      <canvas id='canvasScreen' ref="main"></canvas>
      <p>
        <el-button @click="clear" class='btn btn-default'>clear</el-button>
        <el-button @click="drawInput" class='btn btn-default'>drawInput</el-button>
      </p>
    <div>
      <el-upload class="upload-demo" ref="upload" :auto-upload='false' :on-change='changeUpload'>
        <el-button size="small" type="primary">点击上传</el-button>
        <div slot="tip" class="el-upload__tip">只能上传jpg/png文件，且不超过500kb</div>
      </el-upload>
    </div>
      <p>
        <el-button @click="updateInput" class='btn btn-default'>updateInput</el-button>
      </p>
    </div>
    <div class="child2">
      <el-table :data="tableData" border>
        <el-table-column prop="name" label="name" min-width="20" align="center">
        </el-table-column>
        <el-table-column prop="img" label="img" min-width="40" min-height="170" align="center">
          <template slot-scope="scope">
            <img :src="scope.row.img">
          </template>
        </el-table-column>
        <el-table-column prop="result" label="result" min-width="20" align="center">
        </el-table-column>
        <el-table-column prop="prob" label="prob" min-width="20" align="center">
        </el-table-column>
      </el-table>
    </div>
    <!--<div>-->
    <!--<input id='img_input' type='file' accept='image/*'/>-->
    <!--<div class='preview_box'></div>-->
    <!--<p>-->
    <!--<button @click="drawInput" class='btn btn-default'>mnist_recognize upload</button>-->
    <!--</p>-->
    <!--</div>-->
  </div>
</template>

<script>


  import axios from 'axios'
  import Qs from 'qs'

  function convertImgToBase64(url, callback, outputFormat){
    let canvas = document.createElement('CANVAS'),
    ctx = canvas.getContext('2d'),
    img = new Image;
    img.crossOrigin = 'Anonymous';
    img.onload = function(){
      canvas.height = img.height;
      canvas.width = img.width;
      ctx.drawImage(img,0,0);
      let dataURL = canvas.toDataURL(outputFormat || 'image/png');
      callback.call(this, dataURL);
      canvas = null;
    };
    img.src = url;
  }

  let img_src = '';


  export default {
    name: 'attack',
    data() {
      return {
        isShow: false,
        tableData: [],
      }
    },
    methods: {
      changeUpload: function(file, fileList) {
        this.fileList = fileList;
        this.$nextTick(
          () => {
            let upload_list_li = document.getElementsByClassName('el-upload-list')[0].children;
            for (let i = 0; i < upload_list_li.length; i++) {
              let li_a = upload_list_li[i];
              let imgElement = document.createElement("img");
              let img_url = fileList[i].url;
              convertImgToBase64(img_url, function(base64Img){
                img_src = base64Img;
                imgElement.setAttribute('src', base64Img);
              });
              imgElement.setAttribute('style', "max-width:50%;padding-left:25%");
              // imgElement.setAttribute('style', "max-width:50%;padding-left:25%");
              if (li_a.lastElementChild.nodeName !== 'IMG') {
                li_a.appendChild(imgElement);
              }
            }
          });
      },
      clear: function () {
        console.log('asdf')
        this.ctx = this.$refs.main.getContext('2d')
        this.ctx.fillStyle = '#FFFFFF'
        this.ctx.fillRect(0, 0, 449, 449)
        this.ctx.lineWidth = 1
        this.ctx.strokeRect(0, 0, 449, 449)
        this.ctx.lineWidth = 0.05
        this.tableData = []
        this.$refs.upload.clearFiles()
      },
      drawInput: function () {
        let self = this
        let tdata = []
        let img = new Image()
        img.onload = function () {
          let inputs = []
          const small = document.createElement('canvas').getContext('2d')
          small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28)
          let data = small.getImageData(0, 0, 28, 28).data
          for (let i = 0; i < 28; i++) {
            for (let j = 0; j < 28; j++) {
              let n = 4 * (i * 28 + j)
              inputs[i * 28 + j] = (data[n] + data[n + 1] + data[n + 2]) / 3
            }
          }
          if (Math.min.apply(Math, inputs) === 255) {
            return
          }
          axios.post('/api_mnist/drawInput_defense',
            Qs.stringify({
              inputs: JSON.stringify(inputs)
            })
          )
            .then(function (response) {
              let list = response.data
              for (let _i = 0; _i < response.data['name'].length; _i++) {
                let obj = {}
                obj.name = list.name[_i]
                obj.img = list.img[_i]
                obj.result = list.result[_i]
                obj.prob = list.prob[_i]
                tdata[_i] = obj
              }
              console.log(tdata)
              self.tableData = tdata
            })
        }
        img.src = this.$refs.main.toDataURL()
      },

      updateInput: function () {
        let self = this
        let tdata = []
        let img = new Image()
        img.onload = function () {
          let inputs = []
          const small = document.createElement('canvas').getContext('2d')
          small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28)
          let data = small.getImageData(0, 0, 28, 28).data
          for (let i = 0; i < 28; i++) {
            for (let j = 0; j < 28; j++) {
              let n = 4 * (i * 28 + j)
              inputs[i * 28 + j] = (data[n] + data[n + 1] + data[n + 2]) / 3
            }
          }
          if (Math.min.apply(Math, inputs) === 255) {
            return
          }
          console.log('asdf1')
          axios.post('/api_mnist/drawInput_defense',
            Qs.stringify({
              inputs: JSON.stringify(inputs)
            })
          )
            .then(function (response) {
              let list = response.data
              for (let _i = 0; _i < response.data['name'].length; _i++) {
                let obj = {}
                obj.name = list.name[_i]
                obj.img = list.img[_i]
                obj.result = list.result[_i]
                obj.prob = list.prob[_i]
                tdata[_i] = obj
              }
              console.log(tdata)
              self.tableData = tdata
            })
        }
        img.src = img_src
      }

    },
    mounted() {
      (function () {
        let _createClass = function () {
          function defineProperties(target, props) {
            for (let i = 0; i < props.length; i++) {
              let descriptor = props[i];
              descriptor.enumerable = descriptor.enumerable || false;
              descriptor.configurable = true;
              if ("value" in descriptor) descriptor.writable = true;
              Object.defineProperty(target, descriptor.key, descriptor);
            }
          }

          return function (Constructor, protoProps, staticProps) {
            if (protoProps) defineProperties(Constructor.prototype, protoProps);
            if (staticProps) defineProperties(Constructor, staticProps);
            return Constructor;
          };
        }();

        function _classCallCheck(instance, Constructor) {
          if (!(instance instanceof Constructor)) {
            throw new TypeError("Cannot call a class as a function");
          }
        }
        const Main = function () {
          function Main() {
            _classCallCheck(this, Main)
            this.canvas = document.getElementById('canvasScreen')
            this.input = document.getElementById('input')
            this.update = document.getElementById('update')
            this.canvas.width = 449 // 16 * 28 + 1
            this.canvas.height = 449 // 16 * 28 + 1
            this.ctx = this.canvas.getContext('2d')
            this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this))
            this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this))
            this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this))
          }

          _createClass(Main, [
            {
              key: 'onMouseDown',
              value: function onMouseDown(e) {
                this.canvas.style.cursor = 'default'
                this.drawing = true
                this.prev = this.getPosition(e.clientX, e.clientY)
              }
            }, {
              key: 'onMouseUp',
              value: function onMouseUp() {
                this.drawing = false
                // this.drawInput()
              }
            }, {
              key: 'onMouseMove',
              value: function onMouseMove(e) {
                if (this.drawing) {
                  let curr = this.getPosition(e.clientX, e.clientY)
                  this.ctx.lineWidth = 25
                  this.ctx.lineCap = 'round'
                  this.ctx.beginPath()
                  this.ctx.moveTo(this.prev.x, this.prev.y)
                  this.ctx.lineTo(curr.x, curr.y)
                  this.ctx.stroke()
                  this.ctx.closePath()
                  this.prev = curr
                }
              }
            }, {
              key: 'getPosition',
              value: function getPosition(clientX, clientY) {
                let rect = this.canvas.getBoundingClientRect()
                return {
                  x: clientX - rect.left,
                  y: clientY - rect.top
                }
              }
            }])
          return Main
        }()
        new Main()
      })()
      this.clear()
    }
  }
</script>

<style scoped>
  .parent {
    display: flex;
  }

  .child1 {
    flex: 0.7;
    margin-left: 20px;
  }

  .child2 {
    flex: 1;
    margin-top: 50px;
    margin-right: 20px;
  }
</style>
