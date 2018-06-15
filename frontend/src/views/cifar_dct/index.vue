<template>
  <div class="parent app-container">
    <div class="child1">
      <el-tabs v-model="activeName" style="width: 380px">
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
        <el-table-column prop="name" label="attack_name" min-width="22" align="center">
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
    name: 'cifar_dct',
    data() {
      return {
        isShow: false,
        tableData: [],
        options: [{
          value: '11',
          label: '不选择'
        }, {
          value: '0',
          label: 'airplane'
        }, {
          value: '1',
          label: 'automobile'
        }, {
          value: '2',
          label: 'bird'
        }, {
          value: '3',
          label: 'cat'
        }, {
          value: '4',
          label: 'deer'
        }, {
          value: '5',
          label: 'dog'
        }, {
          value: '6',
          label: 'frog'
        }, {
          value: '7',
          label: 'horse'
        }, {
          value: '8',
          label: 'ship'
        }, {
          value: '9',
          label: 'truck'
        }],
        select_value: '11',
        activeName: 'upload',
        input_fgsm: '0.03',
        input_pgd: '0.03',
        input_bim: '0.03',
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
          small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 32, 32)
          let data = small.getImageData(0, 0, 32, 32).data
          let count = 0
          for (let i = 0; i < 32; i++) {
            for (let j = 0; j < 32; j++) {
              let n = 4 * (i * 32 + j)
              inputs[count++] = data[n]
              inputs[count++] = data[n + 1]
              inputs[count++] = data[n + 2]
            }
          }
          if (Math.min.apply(Math, inputs) === 255) {
            return
          }
          axios.post('/api_cifar/drawInput_cifar_dct',
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
                // obj.attack_prob = list.attack_prob[_i];
                obj.defense_result = list.defense_result[_i];
                obj.echarts = list.echarts[_i];
                obj.echarts_defense = list.echarts_defense[_i]
                tdata[_i] = obj;
              }
              self.tableData = tdata;
            })
        };
        img.src = img_src
      }
    },
    mounted() {
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
