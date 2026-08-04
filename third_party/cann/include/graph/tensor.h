/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GE_TENSOR_H
#define GE_TENSOR_H

#include "graph/buffer.h"
#include "graph/debug/ge_error_codes.h"
#include "graph/attributes_holder.h"
#include "graph/shape.h"
#include "graph/types.h"
#include <atomic>
#include <memory>
#include <vector>

namespace hiai {
class ITensorDescDef;
class ITensorDef;
}

namespace ge {
class Buffer;

class GRAPH_API_EXPORT TensorDesc : public AttrHolder {
    friend class TensorUtils;

public:
    TensorDesc();
    TensorDesc(hiai::ITensorDescDef* tensorDescDef, bool isOwner);
    ~TensorDesc() override;

    explicit TensorDesc(Shape shape, Format format = FORMAT_NCHW, DataType dt = DT_FLOAT);
    explicit TensorDesc(std::vector<Shape> shapes, Format format = FORMAT_NCHW, DataType dt = DT_FLOAT);
    TensorDesc(Shape shape, DataType dt);

    TensorDesc(const TensorDesc& desc);
    TensorDesc& operator=(const TensorDesc& desc);

    GraphErrCodeStatus Save(Buffer& buffer) const;
    GraphErrCodeStatus Load(const uint8_t* data, size_t len);

    const string& GetName() const;
    void SetName(const string& name);

    const Shape& GetShape() const;
    Shape& MutableShape();
    void SetShape(const Shape& shape);

    Shape GetOriginShape() const;
    void SetOriginShape(Shape shape);

    Format GetFormat() const;
    void SetFormat(Format format);

    Format GetOriginFormat() const;
    void SetOriginFormat(Format format);

    DataType GetDataType() const;
    void SetDataType(DataType dataType);

    DataType GetOriginDatatype() const;
    void SetOriginDatatype(DataType dataType);

    bool SerializeTo(hiai::ITensorDescDef* to) const;

protected:
    const hiai::IAttrMapDef* GetAttrMapDef() const override;
    hiai::IAttrMapDef* MutableAttrMapDef() override;

private:
    hiai::ITensorDescDef* tensorDescDef_;
    bool isOwner_;
    Format originFormat_ {FORMAT_RESERVED};
    DataType originDataType_ {DT_UNDEFINED};

    Shape& ShapeReference() const;
    mutable Shape shape_;
    Shape originShape_;

    friend class Tensor;
    void RefTo(const TensorDesc& desc);
};

class GRAPH_API_EXPORT Tensor : public Base {
public:
    Tensor();
    Tensor(hiai::ITensorDef* tensorDef, bool isOwner);
    explicit Tensor(const TensorDesc& tensorDesc);
    Tensor(const TensorDesc& tensorDesc, const Buffer& data);
    Tensor(const TensorDesc& tensorDesc, const uint8_t* data, size_t size);

    ~Tensor() override;

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    std::shared_ptr<Tensor> Clone() const;

    const TensorDesc& GetTensorDesc() const;
    uint32_t GetShapeSize() const;
    TensorDesc& MutableTensorDesc();
    GraphErrCodeStatus SetTensorDesc(const TensorDesc& tensorDesc);

    const Buffer& GetData() const;
    Buffer& MutableData();
    GraphErrCodeStatus SetData(const Buffer& data);
    GraphErrCodeStatus SetData(const uint8_t* data, size_t size);

    bool SerializeTo(hiai::ITensorDef* to) const;

private:
    hiai::ITensorDef* tensorDef_;
    bool isOwner_;

    TensorDesc& DescReference() const;
    mutable TensorDesc desc_;

    Buffer& BufferReference() const;
    mutable Buffer buffer_;
};
} // namespace ge

#endif // GE_TENSOR_H
