/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */
package com.tencent.angel.mlcore


import com.tencent.angel.mlcore.variable.VarState.VarState
import com.tencent.angel.ml.math2.matrix.Matrix
import com.tencent.angel.ml.math2.utils.{DataBlock, LabeledData, RowType}
import com.tencent.angel.ml.math2.vector.Vector
import com.tencent.angel.mlcore.conf.SharedConf
import com.tencent.angel.ml.math2.utils.DataBlock
import com.tencent.angel.mlcore.network.EnvContext
import com.tencent.angel.mlcore.utils.RowTypeUtils
import com.tencent.angel.mlcore.variable.{Variable, VariableManager, VariableProvider}
import org.apache.hadoop.conf.Configuration


abstract class MLModel(val conf: SharedConf) {
  val dataFormat: String = conf.inputDataFormat
  val indexRange: Long = conf.indexRange
  val validIndexNum: Long = conf.modelSize
  val modelType: RowType = conf.modelType
  val isSparseFormat: Boolean = dataFormat == "libsvm" || dataFormat == "dummy"

  //  protected val placeHolder: PlaceHolder
  protected val variableManager: VariableManager
  protected val variableProvider: VariableProvider

  def keyType: String = RowTypeUtils.keyType(modelType)

  def valueType: String = RowTypeUtils.valueType(modelType)

  def storageType: String = RowTypeUtils.storageType(modelType)

  def addVariable(variable: Variable): this.type = {
    variableManager.addVariable(variable)
    this
  }

  def getVariable(name: String): Variable = {
    variableManager.getVariable(name)
  }

  def getAllVariables: List[Variable] = {
    variableManager.getALLVariables
  }

  def hasVariable(v: Variable): Boolean = variableManager.hasVariable(v)

  def hasVariable(name: String): Boolean = variableManager.hasVariable(name)

  def putSlot(v: Variable, g: Matrix): this.type = {
    if (variableManager.hasSlot(v.name)) {
      variableManager.getSlot(v.name).iadd(g)
    } else {
      variableManager.putSlot(v, g)
    }

    this
  }

  def getSlot(name: String): Matrix = {
    variableManager.getSlot(name)
  }

  def getAllSlots: Map[String, Matrix] = {
    variableManager.getAllSlots
  }

  def hasSlot(name: String): Boolean = variableManager.hasSlot(name)

  def putGradient(v: Variable, g: Matrix): this.type = {
    putSlot(v, g)

    this
  }

  def getAllGradients: Map[String, Matrix] = getAllSlots

  def getGradient(name: String): Matrix = getSlot(name)

  def hasGradient(name: String): Boolean = hasSlot(name)

  //  def feedData(data: Array[LabeledData]): this.type = {
  //    placeHolder.asInstanceOf[FeaturePlaceHolder].feedData(data)
  //
  //    this
  //  }

  //---------------------Training Cycle

  /**
    * 创建矩阵
    * @param envCtx
    * @tparam T
    * @return
    */
  def createMatrices[T](envCtx: EnvContext[T]): this.type = {
    variableManager.createALL[T](envCtx)
    this
  }

  def init[T](envCtx: EnvContext[T], taskId: Int = 0): this.type = {
    variableManager.initALL(envCtx, taskId)
    this
  }

  /**
    *
    * @param epoch 迭代次数
    * @param indices 向量索引
    * @return
    */
  def pullParams(epoch: Int, indices: Vector = null): this.type = {
    variableManager.pullALL(epoch, indices)
    this
  }

  /**
    * @param lr 梯度值
    * @return
    */
  def pushSlot(lr: Double): this.type = {
    variableManager.pushALL(lr)
    this
  }

  def update[T](epoch: Int, batchSize: Int): this.type = {
    variableManager.updateALL[T](epoch, batchSize)
    this
  }

  /**
    * 加载模型
    * @param envCtx
    * @param path
    * @param conf
    * @tparam T
    * @return
    */
  def loadModel[T](envCtx: EnvContext[T], path: String, conf: Configuration): this.type = {
    variableManager.loadALL[T](envCtx, path, conf)
    this
  }

  def setState(state: VarState): this.type = {
    variableManager.setAllState(state)
    this
  }

  /**
    * 保存模型
    * @param envCtx
    * @param path
    * @tparam T
    * @return
    */
  def saveModel[T](envCtx: EnvContext[T], path: String): this.type = {
    variableManager.saveALL[T](envCtx, path)
    this
  }

  def releaseMode[T](envCtx: EnvContext[T]): this.type = {
    variableManager.releaseALL(envCtx)

    this
  }

  //---------------------Predict
  /**
    * 根据模型和预测数据得到预测的结果，需要由具体的模型（PSModel）来实现预测流程
    *
    * @param storage DataBlock[LabeledData] 预测数据
    * @return List[PredictResult] 预测结果
    */
  def predict(storage: DataBlock[LabeledData]): List[PredictResult]

  def predict(storage: LabeledData): PredictResult
}
