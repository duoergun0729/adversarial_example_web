<template>
  <div class="parent app-container">
    <div class="child1">
      <el-tabs v-model="activeName" style="width: 380px">
        <el-tab-pane label="画板" name="canvas">
          <canvas id='canvasScreen' ref="main"></canvas>
        </el-tab-pane>
        <el-tab-pane label="上传" name="upload">
          <el-upload
            class="upload-demo"
            action=""
            ref="upload"
            :auto-upload='false'
            :on-change='changeUpload'
            drag
            :limit="1">
            <i class="el-icon-upload"></i>
            <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
            <div slot="tip" class="el-upload__tip">只能上传jpg/png文件</div>
          </el-upload>
        </el-tab-pane>
      </el-tabs>

      <p>
        <span class="span-text">请选择攻击的目标：</span>
        <el-select class="input-width" v-model="select_value" placeholder="请选择">
          <el-option
            v-for="item in options"
            :key="item.value"
            :label="item.label"
            :value="item.value">
          </el-option>
        </el-select>
      </p>
      <p>
        <span class="span-text">请输入FGSM参数：</span>
        <el-input class="input-width"
                  placeholder="请输入内容"
                  v-model="input_fgsm"
                  clearable>
        </el-input>
      </p>
      <p>
        <span class="span-text">请输入PGD参数：</span>
        <el-input class="input-width"
                  placeholder="请输入内容"
                  v-model="input_pgd"
                  clearable>
        </el-input>
      </p>
      <p>
        <span class="span-text">请输入BIM参数：</span>
        <el-input class="input-width"
                  placeholder="请输入内容"
                  v-model="input_bim"
                  clearable>
        </el-input>
      </p>
      <p>
        <el-button @click="clear" class='btn btn-default'>clear</el-button>
        <el-button @click="drawInput" class='btn btn-default'>drawInput</el-button>
      </p>
    </div>
    <div class="child2">
      <el-table :data="tableData">
        <el-table-column type="expand">
          <template slot-scope="props">
            <div class="container">
              <div class="box">
                <pie-chart :chart-data=props.row.echarts></pie-chart>
                <p>attack</p>
              </div>
              <div class="box">
                <pie-chart :chart-data=props.row.echarts_defense></pie-chart>
                <p>defense</p>
              </div>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="name" label="name" min-width="22" align="center">
        </el-table-column>
        <el-table-column prop="img" label="img" min-width="40" min-height="170" align="center">
          <template slot-scope="scope">
            <img :src="scope.row.img">
          </template>
        </el-table-column>
        <el-table-column prop="attack_result" label="attack_result" min-width="22" align="center">
        </el-table-column>
        <el-table-column prop="defense_result" label="defense_result" min-width="22" align="center">
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<script>
  import axios from 'axios'
  import Qs from 'qs'
  import PieChart from './components/PieChart'


  function convertImgToBase64(url, callback, outputFormat) {
    let canvas = document.createElement('CANVAS'),
      ctx = canvas.getContext('2d'),
      img = new Image;
    img.crossOrigin = 'Anonymous';
    img.onload = function () {
      canvas.height = img.height;
      canvas.width = img.width;
      ctx.drawImage(img, 0, 0);
      let dataURL = canvas.toDataURL(outputFormat || 'image/png');
      callback.call(this, dataURL);
      canvas = null;
    };
    img.src = url;
  }

  let img_src = '';

  export default {
    name: 'mnist_dct',
    data() {
      return {
        isShow: false,
        tableData: [],
        options: [{
          value: '11',
          label: '不选择'
        }, {
          value: '0',
          label: '0'
        }, {
          value: '1',
          label: '1'
        }, {
          value: '2',
          label: '2'
        }, {
          value: '3',
          label: '3'
        }, {
          value: '4',
          label: '4'
        }, {
          value: '5',
          label: '5'
        }, {
          value: '6',
          label: '6'
        }, {
          value: '7',
          label: '7'
        }, {
          value: '8',
          label: '8'
        }, {
          value: '9',
          label: '9'
        }],
        select_value: '11',
        activeName: 'canvas',
        input_fgsm: '0.3',
        input_pgd: '0.3',
        input_bim: '0.3',
        echarts: [],
        echarts_defense: [],
      }
    },
    components: {
      PieChart
    },
    methods: {
      changeUpload: function (file, fileList) {
        this.fileList = fileList;
        this.$nextTick(
          () => {
            let upload_list_li = document.getElementsByClassName('el-upload-list')[0].children;
            for (let i = 0; i < upload_list_li.length; i++) {
              let li_a = upload_list_li[i];
              let imgElement = document.createElement("img");
              let img_url = fileList[i].url;
              convertImgToBase64(img_url, function (base64Img) {
                img_src = base64Img;
                imgElement.setAttribute('src', base64Img);
              });
              imgElement.setAttribute('style', "max-width:50%;padding-left:25%");
              if (li_a.lastElementChild.nodeName !== 'IMG') {
                li_a.appendChild(imgElement);
              }
            }
          });
      },
      clear: function () {
        this.ctx = this.$refs.main.getContext('2d')
        this.ctx.fillStyle = '#FFFFFF'
        this.ctx.fillRect(0, 0, 349, 349)
        this.ctx.lineWidth = 1
        this.ctx.strokeRect(0, 0, 349, 349)
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
          axios.post('/api_mnist/drawInput_mnist_dct',
            Qs.stringify({
              inputs: JSON.stringify(inputs),
              target: self.select_value,
              input_fgsm: self.input_fgsm,
              input_pgd: self.input_pgd,
              input_bim: self.input_bim,
            })
          )
            .then(function (response) {
              let list = response.data;
              for (let _i = 0; _i < response.data['name'].length; _i++) {
                let obj = {};
                obj.name = list.name[_i];
                obj.img = list.img[_i];
                obj.attack_result = list.attack_result[_i];
                obj.defense_result = list.defense_result[_i];
                obj.echarts = list.echarts[_i];
                obj.echarts_defense = list.echarts_defense[_i]
                tdata[_i] = obj
              }
              console.log(tdata);
              self.tableData = tdata
            })
        }
        if (this.activeName === 'canvas') {
          img.src = this.$refs.main.toDataURL()
        } else {
          img.src = img_src
        }
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
            this.canvas.width = 349 // 16 * 28 + 1
            this.canvas.height = 349 // 16 * 28 + 1
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
    flex: 0.3;
    margin: 20px 20px;
  }

  .child2 {
    flex: 0.6;
    margin: 50px 20px;
  }

  .span-text {
    display: inline-block;
    width: 150px;
    text-align: right;
  }

  .input-width {
    width: 150px;
  }
  .box {
    width:50%;
    float:left;
    display:inline;
    text-align: center;}
</style>
