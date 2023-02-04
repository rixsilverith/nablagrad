#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <memory>

#include "tensor.hpp"

namespace nabla {
    namespace ta_ops {
        struct TensorOperator {
            std::string name;
            // need to override this
            Tensor forward() { return Tensor({4}); }
            Tensor backward() { return Tensor({4}); }
        };
        struct TensorAdd : public TensorOperator {
            TensorAdd(const Tensor& input0, const Tensor& input1) {
                std::shared_ptr<Tensor> input_0 = std::make_shared<Tensor>(input0);
                std::shared_ptr<Tensor> input_1 = std::make_shared<Tensor>(input1);
                inputs_.push_back(input_0);
                inputs_.push_back(input_1);

                name = "tensor_add_0";
            }

            Tensor forward() {
                Tensor add_tensor(inputs_[0]->shape(), inputs_[0]->requires_grad() || inputs_[1]->requires_grad(), true);
                std::vector<double> res(inputs_[0]->raw_data());
                std::transform(res.begin(), res.end(), inputs_[1]->raw_data().begin(), res.begin(), std::plus<double>());
                add_tensor.setdata(res);
                return add_tensor;
            }

            Tensor backward() {
                return Tensor::ones({2, inputs_[0]->size()});
            }

            const std::vector<std::shared_ptr<Tensor>> inputs() const { return inputs_; }

            std::vector<std::shared_ptr<Tensor>> inputs_;
        };
    } // namespace ta_ops

    namespace autograd {
        struct ComputationGraph;
        struct ComputationNode {
            ComputationNode() {}
            ComputationNode(const Tensor& leaf) : tensor{std::make_shared<Tensor>(leaf)}, is_leaf{tensor->is_leaf_} {}
            ComputationNode(const ta_ops::TensorOperator& t_op) : tensor_op{t_op} {}

            std::shared_ptr<Tensor> tensor;
            bool is_leaf = false;
            ta_ops::TensorOperator tensor_op;
        };

        // A ComputationGraph keep tracks of the operators applied to every tensor requiring
        // gradient computation (i.e. a tensor instantiated with the requires_grad flag)
        // This graph is just a list of 'ComputationNode' objects, which store the actual
        // tensor operator. Leaf nodes, however, are not associated with a tensor operator, but
        // with a leaf tensor (i.e. a tensor with no inputs)
        struct ComputationGraph {
            ComputationGraph(const ComputationGraph&) = delete;

            // Public API of the computation graph. Just calls to the actual internal methods
            static size_t size() { return _instance().size_(); }
            static void push_leaf(Tensor& tensor) { _instance().push_leaf_(tensor); }
            static void push_operator(const ta_ops::TensorOperator& op, Tensor& out) { _instance().push_operator_(op, out); }
            static const ComputationNode& get_operator(size_t op_index) {
                return _instance().get_operator_(op_index);
            }

            static ComputationGraph& _instance() {
                static ComputationGraph cg_instance_;
                return cg_instance_;
            }

        private:
            size_t size_() { return computation_list_.size(); }

            // Push a leaf tensor into the computational graph. Leaf tensors are retained in the
            // graph after gradient backward propagation
            void push_leaf_(Tensor& tensor) {
                tensor.cg_node_idx_ = computation_list_.size();
                tensor.is_leaf_ = true;
                ComputationNode tensor_node(tensor);
                computation_list_.emplace_back(tensor_node);

                std::cout << "[DEBUG] [nabla::autograd::ComputationGraph::push_leaf] Pushed leaf: "
                    << tensor.name() << std::endl;
            }

            void push_operator_(const ta_ops::TensorOperator& op, Tensor& out) {
                out.cg_node_idx_ = computation_list_.size();
                ComputationNode node_op(op);
                computation_list_.emplace_back(node_op);
                std::cout << "[DEBUG] [nabla::autograd::ComputationGraph::push_operator] Pushed operator: "
                   << op.name << std::endl;
            }

            const ComputationNode& get_operator_(size_t op_index) const {
                return computation_list_[op_index];
            }

            ComputationGraph() {}
            std::vector<ComputationNode> computation_list_{};
        };
    } // namespace autograd
} // namespace nabla

#endif // AUTOGRAD_H
